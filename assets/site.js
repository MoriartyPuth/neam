// Neam site shell: header, docs sidebar, prev/next, search, theme, code highlighting and copy.
(function () {
  const body = document.body;
  const ROOT = body.dataset.root || './';
  const PAGE = body.dataset.page || '';

  // One list drives the sidebar, the prev/next links and the search index.
  const NAV = [
    { title: 'Introduction', href: 'docs/' },
    { group: 'Getting started', items: [
      { title: 'Install', href: 'docs/install/' },
      { title: 'Your first prediction', href: 'docs/first-prediction/' },
      { title: 'The output', href: 'docs/output/' },
    ] },
    { group: 'Using Neam', items: [
      { title: 'Articles from a URL', href: 'docs/articles/' },
      { title: 'Long texts', href: 'docs/long-texts/' },
      { title: 'The web app', href: 'docs/web-app/' },
      { title: 'Command line', href: 'docs/command-line/' },
    ] },
    { group: 'How it works', items: [
      { title: 'The pipeline', href: 'docs/pipeline/' },
      { title: 'Eight decisions', href: 'docs/decisions/' },
    ] },
    { group: 'Evaluation', items: [
      { title: 'Scores', href: 'docs/scores/' },
      { title: 'Limitations', href: 'docs/limitations/' },
    ] },
    { group: 'Case studies', items: [
      { title: 'Overview', href: 'case-studies/' },
      { title: 'Reading each word from its meaningful piece', href: 'case-studies/subword-alignment/' },
      { title: 'How Neam reads a Khmer article', href: 'case-studies/reading-an-article/' },
    ] },
    { group: 'Reference', items: [
      { title: 'Cite, license and credits', href: 'docs/cite/' },
    ] },
  ];
  const PAGES = NAV.flatMap(n => n.items ? n.items.map(i => ({ ...i, group: n.group })) : [n]);
  const url = href => ROOT + href;
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const section = PAGE.startsWith('case-studies') ? 'cases' : PAGE.startsWith('docs') ? 'docs' : 'home';

  // ---- theme (shared key with the earlier site)
  const root = document.documentElement;
  try { const t = localStorage.getItem('neam-theme'); if (t) root.dataset.theme = t; } catch (e) {}
  const toggleTheme = () => {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('neam-theme', root.dataset.theme); } catch (e) {}
  };

  // ---- header
  const icons = {
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
    theme: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
    gh: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5a11.5 11.5 0 0 0-3.6 22.4c.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.5-3.9-1.5-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.7 0-1.3.5-2.3 1.2-3.1-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.2 1.2a11 11 0 0 1 5.8 0c2.2-1.5 3.2-1.2 3.2-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.8 1.2 3.1 0 4.4-2.7 5.4-5.3 5.7.4.4.8 1.1.8 2.2v3.2c0 .3.2.7.8.6A11.5 11.5 0 0 0 12 .5z"/></svg>',
    menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
  };
  const mac = /Mac|iPhone|iPad/.test(navigator.platform);
  const header = document.createElement('header');
  header.className = 'site-header';
  header.innerHTML = `<div class="in">
    ${section !== 'home' ? `<button class="iconbtn menubtn" type="button" aria-label="Open the page list">${icons.menu}</button>` : ''}
    <a class="brand" href="${url('')}">Neam <span class="kh" lang="km">នាម</span></a>
    <nav class="topnav" aria-label="Main">
      <a href="${url('docs/')}" ${section === 'docs' ? 'class="on"' : ''}>Docs</a>
      <a href="${url('case-studies/')}" ${section === 'cases' ? 'class="on"' : ''}>Case studies</a>
    </nav>
    <span class="hspace"></span>
    <button class="searchbtn" type="button" aria-label="Search the docs">${icons.search}<span>Search…</span><kbd>${mac ? '⌘' : 'Ctrl'}</kbd><kbd>K</kbd></button>
    <a class="iconbtn" href="https://github.com/MoriartyPuth/neam" aria-label="Neam on GitHub" title="GitHub">${icons.gh}</a>
    <button class="iconbtn" type="button" id="theme" aria-label="Toggle dark mode" title="Toggle dark mode">${icons.theme}</button>
  </div>`;
  body.prepend(header);
  header.querySelector('#theme').addEventListener('click', toggleTheme);
  const mb = header.querySelector('.menubtn');
  if (mb) mb.addEventListener('click', () => body.classList.toggle('nav-open'));
  document.addEventListener('click', e => {
    if (body.classList.contains('nav-open') && !e.target.closest('.sidebar,.menubtn')) body.classList.remove('nav-open');
  });

  // ---- sidebar
  const side = document.getElementById('sidebar');
  if (side) {
    let n = 0;
    side.innerHTML = NAV.map(g => g.items
      ? `<div class="grp"><div class="gh"><i>${++n}</i>${esc(g.group)}</div>${g.items.map(i =>
          `<a href="${url(i.href)}" ${i.href === PAGE ? 'class="on" aria-current="page"' : ''}>${esc(i.title)}</a>`).join('')}</div>`
      : `<div class="grp"><a class="top ${g.href === PAGE ? 'on' : ''}" href="${url(g.href)}">${esc(g.title)}</a></div>`).join('');
  }

  // ---- prev / next
  const pager = document.getElementById('pager');
  const at = PAGES.findIndex(p => p.href === PAGE);
  if (pager && at >= 0) {
    const prev = PAGES[at - 1], next = PAGES[at + 1];
    pager.className = 'pager';
    pager.innerHTML = (prev ? `<a href="${url(prev.href)}"><small>← Previous</small>${esc(prev.title)}</a>` : '<span></span>') +
      (next ? `<a class="next" href="${url(next.href)}"><small>Next →</small>${esc(next.title)}</a>` : '');
  }

  // ---- syntax colouring for <pre data-lang> (python, bash, json)
  const KW = /\b(from|import|for|in|print|def|return|if|else|with|as|True|False|None)\b/;
  function colour(text, lang) {
    const out = [];
    const rules = lang === 'json'
      ? [[/"(?:[^"\\]|\\.)*"(?=\s*:)/y, 't-p'], [/"(?:[^"\\]|\\.)*"/y, 't-s'], [/-?\d+(?:\.\d+)?/y, 't-n'], [/\b(?:true|false|null)\b/y, 't-k']]
      : [[/#[^\n]*/y, 't-c'], [/"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/y, 't-s'], [new RegExp(KW.source, 'y'), 't-k'], [/\b\d+(?:\.\d+)?\b/y, 't-n']];
    let i = 0;
    while (i < text.length) {
      let hit = null;
      for (const [re, cls] of rules) { re.lastIndex = i; const m = re.exec(text); if (m && m[0]) { hit = [m[0], cls]; break; } }
      if (hit) { out.push(`<span class="${hit[1]}">${esc(hit[0])}</span>`); i += hit[0].length; }
      else { out.push(esc(text[i])); i++; }
    }
    return out.join('');
  }
  document.querySelectorAll('pre[data-lang]').forEach(pre => {
    if (pre.dataset.lang !== 'text') pre.innerHTML = colour(pre.textContent, pre.dataset.lang);
  });
  document.querySelectorAll('.code').forEach(box => {
    const btn = box.querySelector('.copy');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const text = box.querySelector('pre').innerText;
      try { await navigator.clipboard.writeText(text); btn.textContent = 'Copied'; }
      catch (e) { btn.textContent = 'Select and copy'; }
      setTimeout(() => (btn.textContent = 'Copy'), 1600);
    });
  });

  // ---- search: fetches the pages once, then matches titles, headings and text
  const dlg = document.createElement('div');
  dlg.className = 'search';
  dlg.setAttribute('role', 'dialog');
  dlg.setAttribute('aria-label', 'Search');
  dlg.innerHTML = '<div class="box"><input type="search" placeholder="Search the docs…" aria-label="Search the docs"><ul></ul></div>';
  body.append(dlg);
  const input = dlg.querySelector('input'), list = dlg.querySelector('ul');
  let index = null, building = null, sel = 0;
  function build() {
    return building || (building = buildIndex());
  }
  async function buildIndex() {
    const pages = [{ title: 'Home', href: '', group: '', heads: [], text: 'Neam Khmer named entity recognition' }];
    await Promise.all(PAGES.map(async p => {
      try {
        const html = await (await fetch(url(p.href))).text();
        const d = new DOMParser().parseFromString(html, 'text/html');
        d.querySelectorAll('script,style').forEach(n => n.remove());
        const main = d.querySelector('main') || d.body;
        main.querySelectorAll('p,li,td,th,h1,h2,h3,b,pre,div').forEach(e => e.append(' '));   // keep words apart in snippets
        pages.push({ ...p, heads: [...main.querySelectorAll('h2,h3')].map(h => ({ t: h.textContent.trim(), id: h.id })), text: main.textContent.replace(/\s+/g, ' ') });
      } catch (e) { pages.push({ ...p, heads: [], text: '' }); }
    }));
    return (index = pages);
  }
  async function run() {
    const q = input.value.trim().toLowerCase();
    await build();
    if (q !== input.value.trim().toLowerCase()) return;   // a newer keystroke will render
    const hits = [];
    if (q) for (const p of index) {
      if (p.title.toLowerCase().includes(q)) hits.push({ href: p.href, title: p.title, sub: p.group || 'Page', w: 3 });
      for (const h of p.heads) if (h.t.toLowerCase().includes(q)) hits.push({ href: p.href + (h.id ? '#' + h.id : ''), title: h.t, sub: p.title, w: 2 });
      const k = p.text.toLowerCase().indexOf(q);
      if (k >= 0 && !hits.some(h => h.href.split('#')[0] === p.href)) hits.push({ href: p.href, title: p.title, sub: '…' + p.text.slice(Math.max(0, k - 40), k + 70).trim() + '…', w: 1 });
    }
    hits.sort((a, b) => b.w - a.w);
    sel = 0;
    list.innerHTML = q
      ? (hits.length ? hits.slice(0, 12).map((h, i) => `<li><a href="${url(h.href)}" class="${i ? '' : 'sel'}">${esc(h.title)}<small>${esc(h.sub)}</small></a></li>`).join('') : '<li class="empty">No results.</li>')
      : PAGES.slice(0, 6).map((p, i) => `<li><a href="${url(p.href)}" class="${i ? '' : 'sel'}">${esc(p.title)}<small>${esc(p.group || 'Docs')}</small></a></li>`).join('');
  }
  const open = () => { dlg.classList.add('open'); input.value = ''; run(); setTimeout(() => input.focus(), 0); };
  const close = () => dlg.classList.remove('open');
  header.querySelector('.searchbtn').addEventListener('click', open);
  dlg.addEventListener('click', e => { if (e.target === dlg) close(); });
  input.addEventListener('input', run);
  input.addEventListener('keydown', e => {
    const links = [...list.querySelectorAll('a')];
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      sel = (sel + (e.key === 'ArrowDown' ? 1 : -1) + links.length) % Math.max(links.length, 1);
      links.forEach((a, i) => a.classList.toggle('sel', i === sel));
    } else if (e.key === 'Enter' && links[sel]) location.href = links[sel].href;
  });
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); dlg.classList.contains('open') ? close() : open(); }
    else if (e.key === 'Escape') { close(); body.classList.remove('nav-open'); }
  });
})();
