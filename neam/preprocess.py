"""Lossless Khmer segmentation with original offsets, and word-aligned subword windows."""
from dataclasses import dataclass
import unicodedata

LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
HIDDEN = frozenset('\u200b\u200c\u200d')
# Which subword of each word carries its label. SentencePiece emits a bare word
# marker before many pre-split Khmer words, so the first piece is often an
# identical "\u2581" token with no lexical content. Checkpoints without a recorded
# position were trained on the first piece and must keep using it.
FIRST_SUBWORD = 'first-subword'
FIRST_CONTENT_SUBWORD = 'first-content-subword'
LABEL_POSITIONS = (FIRST_SUBWORD, FIRST_CONTENT_SUBWORD)
_WORD_MARKERS = '\u2581'


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int


def normalize(text):
    """Remove only selected format characters; retain coeng and combining marks."""
    return ''.join(c for c in text if c not in HIDDEN)


def _split_edge_punctuation(piece):
    """Separate punctuation attached to a word by the external segmenter."""
    left = 0
    while left < len(piece) and unicodedata.category(piece[left]).startswith('P'):
        left += 1
    right = len(piece)
    while right > left and unicodedata.category(piece[right - 1]).startswith('P'):
        right -= 1
    return ([*piece[:left]] + ([piece[left:right]] if left < right else [])
            + [*piece[right:]])


def segment(text, segmenter=None):
    if segmenter is None:
        try:
            from khmer_nltk import word_tokenize
        except ModuleNotFoundError as exc:
            if exc.name != 'khmer_nltk':
                raise
            # The PyPI khmer-nltk 1.6 wheel exports khmernltk, without an underscore.
            from khmernltk import word_tokenize
        segmenter = lambda value: word_tokenize(value, return_tokens=True)
    positions = [i for i, c in enumerate(text) if c not in HIDDEN]
    clean = normalize(text)
    pieces = segmenter(clean) if clean.strip() else []
    if isinstance(pieces, str):
        raise ValueError('Segmenter must return a list of words, not a string.')
    pieces = [part for piece in pieces for part in _split_edge_punctuation(piece)]
    words, cursor = [], 0
    for piece in pieces:
        if not piece or piece.isspace():
            continue
        start = clean.find(piece, cursor)
        if start < 0 or clean[cursor:start].strip():
            raise ValueError('Segmentation changed or skipped non-whitespace text.')
        end = start + len(piece)
        words.append(Word(piece, positions[start], positions[end - 1] + 1))
        cursor = end
    if clean[cursor:].strip():
        raise ValueError('Segmentation omitted trailing text.')
    return words


def segment_document(text, segmenter=None):
    """Segment multiline inference text while preserving document offsets."""
    words, offset = [], 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip('\r\n\v\f\x85\u2028\u2029')
        words.extend(Word(word.text, word.start + offset, word.end + offset)
                     for word in segment(content, segmenter))
        offset += len(line)
    if not text or offset == len(text):
        return words
    # ``splitlines`` normally consumes the full string; retain a defensive tail.
    words.extend(Word(word.text, word.start + offset, word.end + offset)
                 for word in segment(text[offset:], segmenter))
    return words


def validate_tags(tags):
    previous = 'O'
    for tag in tags:
        if tag not in LABEL2ID:
            raise ValueError(f'Unknown IOB2 tag: {tag!r}')
        if tag.startswith('I-') and previous not in ('B-' + tag[2:], tag):
            raise ValueError(f'Invalid IOB2 transition: {previous} -> {tag}')
        previous = tag


def spans_to_words(text, entities, segmenter=None):
    # Gold data may contain full articles. Segment line by line so newline and
    # paragraph boundaries cannot be rewritten by the third-party segmenter.
    words = segment_document(text, segmenter)
    ordered = sorted(entities, key=lambda e: e['start'])
    previous_end = 0
    boundaries = set()
    for entity in ordered:
        start, end, label = entity['start'], entity['end'], entity['label']
        if (not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end <= len(text) or start < previous_end
                or label not in ('PER', 'ORG', 'LOC')):
            raise ValueError(f'Invalid or overlapping entity: {entity}')
        boundaries.update((start, end))
        previous_end = end
    # Khmer segmentation is statistical and can occasionally join an entity to
    # the next unspaced word. Reviewed character boundaries take precedence.
    split_words = []
    for word in words:
        cuts = [word.start] + sorted(
            boundary for boundary in boundaries if word.start < boundary < word.end
        ) + [word.end]
        for start, end in zip(cuts, cuts[1:]):
            value = normalize(text[start:end])
            if value:
                split_words.append(Word(value, start, end))
    words = split_words
    tags = ['O'] * len(words)
    previous_end = 0
    for entity in ordered:
        start, end, label = entity['start'], entity['end'], entity['label']
        selected = [i for i, w in enumerate(words) if w.start < end and w.end > start]
        if not selected or words[selected[0]].start != start or words[selected[-1]].end != end:
            raise ValueError(f'Entity boundary conflicts with segmentation: {entity}')
        for j, i in enumerate(selected):
            tags[i] = ('B-' if j == 0 else 'I-') + label
        previous_end = end
    return [w.text for w in words], tags


def word_label_positions(word_ids, tokens=None, label_position=FIRST_SUBWORD):
    """Return {word index: token position} for the subword that carries each label."""
    if label_position not in LABEL_POSITIONS:
        raise ValueError(f'Unknown label position: {label_position!r}')
    if label_position == FIRST_CONTENT_SUBWORD and tokens is None:
        raise ValueError('Token strings are required for first-content-subword alignment.')
    positions, content = {}, set()
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in content:
            continue
        if word_id not in positions:
            positions[word_id] = position
        if label_position == FIRST_SUBWORD:
            content.add(word_id)
        elif tokens[position].strip(_WORD_MARKERS):
            # Replace a bare marker with the word's first lexical piece. A word
            # consisting only of markers keeps its first piece.
            positions[word_id] = position
            content.add(word_id)
    return positions


def align_labels(word_ids, tags, tokens=None, label_position=FIRST_SUBWORD):
    """Special tokens and every unselected subword have ignore index -100."""
    positions = word_label_positions(word_ids, tokens, label_position)
    labels = [-100] * len(word_ids)
    for word_id, position in positions.items():
        labels[position] = LABEL2ID[tags[word_id]]
    return labels


def _check_tokenizer(tokenizer, max_length):
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required for word_ids alignment.')
    if max_length < 8 or max_length > tokenizer.model_max_length:
        raise ValueError('max_length must be >= 8 and within tokenizer limits.')


def _word_piece_counts(tokenizer, words):
    """Tokenize the complete word list once to get exact per-word piece counts."""
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False,
                        truncation=False, verbose=False)
    counts = [0] * len(words)
    for wid in encoded.word_ids():
        if wid is not None:
            counts[wid] += 1
    if any(c == 0 for c in counts):
        raise ValueError('Tokenizer dropped a segmented word.')
    return counts


def _encode_window(tokenizer, words, start, end, max_length):
    chunk = tokenizer(words[start:end], is_split_into_words=True, truncation=False)
    if len(chunk['input_ids']) > max_length:
        raise ValueError('Tokenizer produced inconsistent word lengths.')
    ids = chunk.word_ids()
    if set(w for w in ids if w is not None) != set(range(end - start)):
        raise ValueError('Tokenizer dropped a word within a chunk.')
    return chunk, ids


def encode_chunks(tokenizer, words, tags=None, max_length=256, label_position=FIRST_SUBWORD):
    """Pack whole words, never truncate subwords or split a gold entity."""
    _check_tokenizer(tokenizer, max_length)
    if tags is not None:
        if len(tags) != len(words):
            raise ValueError('Word/tag count mismatch.')
        validate_tags(tags)
    if not words:
        return []
    counts = _word_piece_counts(tokenizer, words)
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    chunks, start = [], 0
    while start < len(words):
        end, used = start, 0
        while end < len(words) and used + counts[end] <= budget:
            used += counts[end]
            end += 1
        if tags is not None:
            while end < len(words) and end > start and tags[end].startswith('I-'):
                end -= 1
        if end == start:
            raise ValueError('A word or gold entity exceeds max_length; increase it.')
        chunk, ids = _encode_window(tokenizer, words, start, end, max_length)
        if tags is not None:
            chunk['labels'] = align_labels(ids, tags[start:end], chunk.tokens(), label_position)
        chunks.append((start, end, chunk, ids))
        start = end
    return chunks


def encode_context_windows(tokenizer, words, max_length=256):
    """Overlapping whole-word inference windows; each word is read where it has most context.

    Windows advance by about half their subword budget, and the final window is
    right-aligned to stay full. Each word is owned by the window that maximizes the
    smaller of its left and right subword context; a document edge counts as full
    context. Returns (start, end, chunk, word_ids, owned word indices) tuples.
    """
    _check_tokenizer(tokenizer, max_length)
    if not words:
        return []
    counts = _word_piece_counts(tokenizer, words)
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if max(counts) > budget:
        raise ValueError('A word exceeds max_length; increase it.')
    total = len(words)
    spans, start = [], 0
    while True:
        end, used = start, 0
        while end < total and used + counts[end] <= budget:
            used += counts[end]
            end += 1
        if end == total and start > 0:
            used = sum(counts[start:end])
            while start > 0 and used + counts[start - 1] <= budget:
                start -= 1
                used += counts[start]
        if not spans or spans[-1] != (start, end):
            spans.append((start, end))
        if end == total:
            break
        advance, next_start = 0, start
        while next_start < end and advance + counts[next_start] <= used / 2:
            advance += counts[next_start]
            next_start += 1
        start = max(next_start, start + 1)
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + count)
    infinite = float('inf')
    owned = {span: [] for span in spans}
    for word in range(total):
        best, best_margin = None, -1.0
        for window_start, window_end in spans:
            if not window_start <= word < window_end:
                continue
            left = prefix[word] - prefix[window_start] if window_start > 0 else infinite
            right = prefix[window_end] - prefix[word + 1] if window_end < total else infinite
            margin = min(left, right)
            if margin > best_margin:
                best, best_margin = (window_start, window_end), margin
        owned[best].append(word)
    windows = []
    for window_start, window_end in spans:
        if owned[(window_start, window_end)]:
            chunk, ids = _encode_window(tokenizer, words, window_start, window_end, max_length)
            windows.append((window_start, window_end, chunk, ids, owned[(window_start, window_end)]))
    return windows
