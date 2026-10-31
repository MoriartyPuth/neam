"""Safe text extraction from public web articles."""
import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

MAX_INPUT_CHARS = 20_000
MAX_WEB_BYTES = 5 * 1024 * 1024
ARTICLE_UI_LINES = frozenset({'Share', 'Tweet', 'Print', 'Facebook', 'Telegram'})
_KHMER = re.compile('[ក-៿]')


def trim_text(text, max_chars=MAX_INPUT_CHARS):
    text = text.strip()
    return text[:max_chars], len(text) > max_chars


def _validate_public_url(url):
    parsed = urlparse(url.strip())
    if parsed.scheme not in ('http', 'https') or not parsed.hostname:
        raise ValueError('Enter a complete http:// or https:// article URL.')
    if parsed.username or parsed.password:
        raise ValueError('URLs containing credentials are not supported.')
    try:
        addresses = {entry[4][0] for entry in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise ValueError('The article hostname could not be resolved.') from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError('Only public internet article URLs are supported.')
    return parsed.geturl()


def article_text_from_html(html):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    for node in soup.select(
            'script, style, nav, header, footer, aside, form, noscript, svg, '
            'blockquote.wp-embedded-content'):
        node.decompose()
    # Prefer explicit body containers over an outer <article>, which often also
    # contains navigation, share controls, and related-story cards.
    selectors = (
        '[itemprop="articleBody"]', '.post_details_block .adaptive', '#mvp-content-main',
        '.article-content', '.entry-content',
        '.post-content', '.story-body', '.content-detail', '.td-post-content',
        '.tdb_single_content', '.prose', 'div.content', 'article',
    )
    root = None
    for selector in selectors:
        candidates = [node for node in soup.select(selector)
                      if len(node.get_text(' ', strip=True)) >= 200]
        if candidates:
            root = max(candidates, key=lambda node: len(node.get_text(' ', strip=True)))
            break
    root = root or soup.find('main') or soup.find(attrs={'role': 'main'}) or soup.body or soup
    lines = [re.sub(r'\s+', ' ', line).strip() for line in root.get_text('\n').splitlines()]
    return '\n'.join(trim_non_khmer_edges(
        [line for line in lines if line and line not in ARTICLE_UI_LINES]))


def trim_non_khmer_edges(lines):
    """On a Khmer page, drop the lines before the first and after the last Khmer line.

    Those are site menus and footers. Non-Khmer lines between Khmer lines are kept,
    because some sites put a foreign name or a date on its own line.
    """
    khmer = [i for i, line in enumerate(lines) if _KHMER.search(line)]
    return lines[khmer[0]:khmer[-1] + 1] if khmer else lines


def extract_article_text(url, max_chars=MAX_INPUT_CHARS):
    import requests

    current = url.strip()
    response = None
    for _ in range(5):
        current = _validate_public_url(current)
        try:
            response = requests.get(
                current, headers={'User-Agent': 'KhmerNER/1.0 article text extractor'},
                timeout=(5, 20), allow_redirects=False, stream=True)
        except requests.RequestException as exc:
            raise ValueError(f'Could not download the article: {exc}.') from exc
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get('location')
            response.close()
            if not location:
                raise ValueError('The article returned an invalid redirect.')
            current = urljoin(current, location)
            continue
        break
    else:
        raise ValueError('The article redirected too many times.')
    try:
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f'Article request failed: {exc}.') from exc
        content_type = response.headers.get('content-type', '').lower()
        if not any(kind in content_type for kind in ('text/html', 'application/xhtml+xml', 'text/plain')):
            raise ValueError(f'URL returned unsupported content type: {content_type or "unknown"}.')
        declared = int(response.headers.get('content-length', '0') or 0)
        if declared > MAX_WEB_BYTES:
            raise ValueError('Article download exceeds the 5 MB limit.')
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_WEB_BYTES:
                raise ValueError('Article download exceeds the 5 MB limit.')
            chunks.append(chunk)
        encoding = response.encoding or 'utf-8'
        html = b''.join(chunks).decode(encoding, errors='replace')
    finally:
        response.close()
    text = article_text_from_html(html) if 'html' in content_type else html
    text, truncated = trim_text(text, max_chars)
    if not text:
        raise ValueError('No readable article text was found at this URL.')
    status = f'Loaded {len(text):,} characters from the article.'
    if truncated:
        status += f' Content was limited to the first {max_chars:,} characters.'
    return text, status
