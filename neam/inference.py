"""Word-aligned Khmer NER inference: ensembles, Viterbi decoding, overlapping windows."""
import argparse
import json
import os
from pathlib import Path

import regex

from neam.preprocess import (FIRST_SUBWORD, LABELS, encode_chunks, encode_context_windows,
                              segment_document, word_label_positions)

# Only short conventional honorifics; ceremonial titles stay model decisions.
PERSON_PREFIXES = frozenset({'លោក', 'លោកស្រី', 'អ្នកស្រី', 'កញ្ញា'})
LOCATION_PREFIXES = frozenset({
    'រាជធានី', 'ខេត្ត', 'ក្រុង', 'ស្រុក', 'ឃុំ', 'សង្កាត់', 'ភូមិ', 'ប្រទេស',
    'ប្រាសាទ', 'ទន្លេ', 'ភ្នំ', 'កោះ', 'បឹង',
})
BOUNDARY_PUNCTUATION = frozenset({
    '។', '៖', '៕', '៘', '៙', '៚', '.', ',', '،', ';', ':', '!', '?', '(', ')', '«', '»',
})
# Periods, commas and brackets can belong to a predicted name or abbreviation.
SENTENCE_PUNCTUATION = frozenset({'។', '៖', '៕', '៘', '៙', '៚', ';', ':', '!', '?'})
_DOTTED_INITIALISM = regex.compile(r'(?:\p{L}\p{M}*\.){2,}$')
_CLOSING_PAIRS = {')': '(', '»': '«'}
RECONSTRUCTION_VERSION = 'model-punctuation-short-honorifics-nationality-titles-whitespace-v4'
# Hugging Face model ID used when no local model folder is given.
DEFAULT_MODEL = 'MoriartyPuth/neam'
# A model directory holding this file averages its members' word probabilities.
ENSEMBLE_FILE = 'ensemble.json'
ENSEMBLE_SETTINGS = {
    'o_bias': 'khmer_ner_o_bias', 'label_biases': 'khmer_ner_label_biases',
    'expand_display_boundaries': 'khmer_ner_expand_display_boundaries',
    'decoding': 'khmer_ner_decoding',
    'context_windows': 'khmer_ner_context_windows',
}
# disjoint: back-to-back windows; overlap: each word read in its most central window.
CONTEXT_WINDOWS = ('disjoint', 'overlap')
# "ជនជាតិ X" names a nationality or people group, which is O under the three-label
# policy; X alone as a place stays LOC. Interim display rule until training data
# teaches the distinction; strict model tags are unchanged.
NATIONALITY_MARKERS = frozenset({'ជនជាតិ'})
# Ceremonial titles and ranks stay outside PER spans under the annotation policy;
# short honorifics such as លោក remain part of the name.
CEREMONIAL_TITLES = frozenset({
    'ឯកឧត្តម', 'ឯកឧត្ដម', 'លោកជំទាវ', 'សម្តេច', 'សម្ដេច', 'ឧកញ៉ា', 'អ្នកឧកញ៉ា', 'លោកបណ្ឌិត', 'បណ្ឌិត',
})
_RANK_MARKER = 'សេនីយ៍'  # military ranks: វរសេនីយ៍ឯក, ឧត្តមសេនីយ៍ទោ, ...
# Parts that only continue a title (ឯកឧត្ដម សន្ដិ បណ្ឌិត); alone they can be part of a name.
TITLE_CONTINUATIONS = frozenset({'សន្តិ', 'សន្ដិ'})
_ZERO_WIDTH = frozenset({'\u200b', '\u200c', '\u200d'})
PERSON_COORDINATION_FOLLOWERS = frozenset({
    'ដែល', 'បាន', 'កំពុង', 'នឹង', 'រស់', 'ជា', 'មកពី', 'ទៅ', 'នៅ',
}) | BOUNDARY_PUNCTUATION


def repair_iob2(tags):
    result, previous = [], 'O'
    for tag in tags:
        if tag.startswith('I-') and previous not in ('B-' + tag[2:], tag):
            tag = 'B-' + tag[2:]
        result.append(tag)
        previous = tag
    return result


DECODINGS = ('argmax', 'viterbi')


def _allowed_transition(previous, tag):
    """IOB2: an I-tag must continue an entity of the same type."""
    return not tag.startswith('I-') or previous in ('B-' + tag[2:], tag)


def _viterbi(rows, labels):
    """Highest-scoring tag sequence with no invalid IOB2 transitions."""
    import math
    logs = [[math.log(max(value, 1e-12)) for value in row] for row in rows]
    best = [score if _allowed_transition('O', labels[j]) else float('-inf')
            for j, score in enumerate(logs[0])]
    back = []
    for row in logs[1:]:
        pointers, scores = [], []
        for j, tag in enumerate(labels):
            candidates = [(best[i], i) for i, previous in enumerate(labels)
                          if _allowed_transition(previous, tag)]
            score, index = max(candidates)
            scores.append(score + row[j])
            pointers.append(index)
        best = scores
        back.append(pointers)
    path = [max(range(len(labels)), key=best.__getitem__)]
    for pointers in reversed(back):
        path.append(pointers[path[-1]])
    return path[::-1]


def decode_probabilities(probabilities, id2label, o_bias=1.0, repair=True, label_biases=None,
                         decoding='argmax'):
    """Decode word probabilities after applying a validation-calibrated O bias.

    ``argmax`` picks each word's best label and repairs orphan I-tags. ``viterbi``
    picks the best whole sequence among valid IOB2 paths over the same biased scores.
    """
    if o_bias <= 0:
        raise ValueError('o_bias must be positive.')
    if decoding not in DECODINGS:
        raise ValueError(f'Unknown decoding: {decoding!r}')
    o_id = next((idx for idx, label in id2label.items() if label == 'O'), None)
    if o_id is None:
        raise ValueError('Label vocabulary does not contain O.')
    label_biases = label_biases or {}
    adjusted_rows = []
    for row in probabilities:
        if len(row) != len(id2label):
            raise ValueError('Probability row does not match the label vocabulary.')
        adjusted = list(row)
        adjusted[o_id] *= o_bias
        for label_id, label in id2label.items():
            adjusted[label_id] *= float(label_biases.get(label, 1.0))
        adjusted_rows.append(adjusted)
    if decoding == 'viterbi' and adjusted_rows:
        ids = sorted(id2label)
        label_ids = [ids[j] for j in _viterbi(
            [[row[i] for i in ids] for row in adjusted_rows], [id2label[i] for i in ids])]
    else:
        label_ids = [max(range(len(row)), key=row.__getitem__) for row in adjusted_rows]
    tags = [id2label[label_id] for label_id in label_ids]
    scores = [float(row[label_id]) for row, label_id in zip(probabilities, label_ids)]
    return (repair_iob2(tags) if repair else tags), scores


def expand_entity_boundaries(words, tags):
    """Complete common Khmer named-entity designators for display spans."""
    if len(words) != len(tags):
        raise ValueError('Word/tag lengths differ.')
    result = list(tags)
    prefixes = {'PER': PERSON_PREFIXES, 'LOC': LOCATION_PREFIXES}
    for i in range(len(result) - 1):
        if (words[i] in PERSON_PREFIXES and result[i] == 'B-PER'
                and result[i + 1] == 'B-PER'):
            result[i + 1] = 'I-PER'
    for i, tag in enumerate(list(result)):
        if not tag.startswith('B-') or i == 0:
            continue
        entity_type = tag[2:]
        if result[i - 1] == 'O' and words[i - 1] in prefixes.get(entity_type, ()):
            result[i - 1] = 'B-' + entity_type
            result[i] = 'I-' + entity_type

    # Khmer personal names do not have capitalization. Complete a conservative
    # coordinated-name pattern when the following token begins a new clause.
    for i in range(1, len(result) - 1):
        if (words[i] == 'និង' and result[i] == 'O'
                and result[i - 1].endswith('-PER') and result[i + 1] == 'O'
                and (i + 2 == len(result) or words[i + 2] in PERSON_COORDINATION_FOLLOWERS)):
            result[i + 1] = 'B-PER'

    return repair_iob2(result)


def clear_nationality_locations(words, tags):
    """Untag a LOC entity that is a nationality phrase (marker + country)."""
    result = list(tags)
    i = 0
    while i < len(result):
        starts_after_marker = i > 0 and words[i - 1] in NATIONALITY_MARKERS and result[i - 1] == 'O'
        if result[i] == 'B-LOC' and (starts_after_marker or words[i] in NATIONALITY_MARKERS):
            result[i] = 'O'
            i += 1
            while i < len(result) and result[i] == 'I-LOC':
                result[i] = 'O'
                i += 1
            continue
        i += 1
    return result


def _is_ceremonial_title(word):
    return word in CEREMONIAL_TITLES or word.startswith(('សម្តេច', 'សម្ដេច')) or _RANK_MARKER in word


def strip_ceremonial_titles(words, tags):
    """Move a PER start past leading ceremonial titles when a name follows them."""
    result = list(tags)
    for i, tag in enumerate(tags):
        if tag != 'B-PER' or not _is_ceremonial_title(words[i]):
            continue
        j = i
        while (j < len(result) and result[j].endswith('-PER')
               and (_is_ceremonial_title(words[j]) or words[j] in TITLE_CONTINUATIONS
                    or not words[j].strip())):
            j += 1
        if j < len(result) and result[j] == 'I-PER':  # a name remains after the titles
            for k in range(i, j):
                result[k] = 'O'
            result[j] = 'B-PER'
    return result


def prepare_display_tags(words, model_tags):
    """Keep model-labeled name punctuation, drop nationality LOCs, complete short prefixes."""
    if len(words) != len(model_tags):
        raise ValueError('Word/tag lengths differ.')
    cleaned = ['O' if word in SENTENCE_PUNCTUATION else tag
               for word, tag in zip(words, model_tags)]
    expanded = expand_entity_boundaries(words, clear_nationality_locations(words, cleaned))
    return strip_ceremonial_titles(words, expanded)


def _outside_span(character):
    """Characters never kept at a span edge: punctuation, whitespace, zero-width marks."""
    return character in BOUNDARY_PUNCTUATION or character.isspace() or character in _ZERO_WIDTH


def _name_end_punctuation(surface):
    """Whether the final mark is structural within this predicted name."""
    if surface.endswith('.'):
        return bool(_DOTTED_INITIALISM.search(surface))
    closing = surface[-1:]
    opening = _CLOSING_PAIRS.get(closing)
    if opening:
        depth = 0
        for character in surface:
            if character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if depth < 0:
                    return False
        return opening in surface and depth == 0
    return False


def collect_entities(text, words, tags, scores, model_tags=None):
    entities = []
    active = None
    model_tags = tags if model_tags is None else model_tags
    for word, tag, score, model_tag in zip(words, repair_iob2(tags), scores, model_tags):
        if tag == 'O' or tag.startswith('B-'):
            active = None
        elif active is not None and '\n' in text[active['end']:word.end]:
            # A name never continues onto the next line (menus, lists, captions).
            active = None
            if not word.text.strip():
                continue
        if tag == 'O':
            continue
        if active is None:
            active = {'start': word.start, 'end': word.end, 'label': tag[2:],
                      '_scores': [], '_boundary_adjusted': False}
            entities.append(active)
        active['end'] = word.end
        if model_tag.endswith('-' + active['label']):
            active['_scores'].append(score)
        if model_tag != tag:
            active['_boundary_adjusted'] = True
    completed = []
    for entity in entities:
        # Khmer punctuation is sometimes attached to a lexical token by the
        # segmenter (for example, "សុវិជាតិ៖"). Keep it outside the entity span.
        while (entity['start'] < entity['end']
               and _outside_span(text[entity['start']])):
            entity['start'] += 1
            entity['_boundary_adjusted'] = True
        while (entity['end'] > entity['start']
               and _outside_span(text[entity['end'] - 1])):
            if (text[entity['end'] - 1] in BOUNDARY_PUNCTUATION
                    and _name_end_punctuation(text[entity['start']:entity['end']])):
                break
            entity['end'] -= 1
            entity['_boundary_adjusted'] = True
        if entity['start'] == entity['end']:
            continue  # A punctuation-only prediction is not a named entity.
        entity['text'] = text[entity['start']:entity['end']]
        entity['score'] = sum(entity['_scores']) / len(entity['_scores']) if entity['_scores'] else 0.0
        entity.pop('_scores')
        adjusted = entity.pop('_boundary_adjusted')
        if adjusted:
            entity['boundary_adjusted'] = True
        completed.append(entity)
    return completed


def load_ensemble_spec(model_path):
    """Return member paths and setting overrides; a plain checkpoint is one member."""
    manifest = Path(model_path) / ENSEMBLE_FILE
    if not manifest.exists():
        return [str(model_path)], {}
    spec = json.loads(manifest.read_text(encoding='utf-8'))
    members = [os.path.normpath(member if Path(member).is_absolute() else Path(model_path) / member)
               for member in spec.get('members', [])]
    if not members:
        raise ValueError(f'{manifest} lists no members.')
    unknown = set(spec) - {'members', 'note'} - set(ENSEMBLE_SETTINGS)
    if unknown:
        raise ValueError(f'{manifest} has unknown keys: {sorted(unknown)}')
    return members, {key: spec[key] for key in ENSEMBLE_SETTINGS if key in spec}


def resolve_model_path(model=DEFAULT_MODEL, revision=None):
    """Return a local folder: the path itself, or a Hugging Face Hub snapshot of the model ID."""
    if Path(model).exists():
        return str(model)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model, revision=revision)


class Neam:
    """Khmer PER/ORG/LOC tagger; ``predict(text)`` returns entities with original offsets."""

    @classmethod
    def from_pretrained(cls, model=DEFAULT_MODEL, **kwargs):
        return cls(model, **kwargs)

    def __init__(self, model=DEFAULT_MODEL, device=None, max_length=None, revision=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        model_path = resolve_model_path(model, revision)
        members, overrides = load_ensemble_spec(model_path)
        self.ensemble = len(members) > 1 or bool(overrides)
        # Keep the exact SentencePiece behavior used for training. Passing False also
        # avoids a Transformers 4.57.6 false-positive Mistral-regex warning for XLM-R.
        self.tokenizer = AutoTokenizer.from_pretrained(
            members[0], use_fast=True, fix_mistral_regex=False)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.models = [AutoModelForTokenClassification.from_pretrained(member).to(self.device).eval()
                       for member in members]
        self.model = self.models[0]
        config = self.model.config
        self.id2label = {int(k): v for k, v in config.id2label.items()}
        if set(self.id2label.values()) != set(LABELS) or len(self.id2label) != len(LABELS):
            raise ValueError('Model must be fine-tuned with the seven PER/ORG/LOC IOB2 labels.')
        self.max_length = max_length or getattr(config, 'khmer_ner_max_length', 256)
        # Older checkpoints record no position and were trained on the first piece.
        self.label_position = getattr(config, 'khmer_ner_label_position', FIRST_SUBWORD)
        for member, model in zip(members[1:], self.models[1:]):
            if ({int(k): v for k, v in model.config.id2label.items()} != self.id2label
                    or getattr(model.config, 'khmer_ner_label_position', FIRST_SUBWORD) != self.label_position
                    or model.config.vocab_size != config.vocab_size):
                raise ValueError(f'Ensemble member {member} does not match {members[0]}.')
        settings = {key: getattr(config, name, None) for key, name in ENSEMBLE_SETTINGS.items()}
        settings.update(overrides)
        self.o_bias = float(settings['o_bias'] if settings['o_bias'] is not None else 1.0)
        self.label_biases = dict(settings['label_biases'] or {})
        self.expand_display_boundaries = bool(settings['expand_display_boundaries'])
        self.decoding = settings['decoding'] or 'argmax'
        if self.decoding not in DECODINGS:
            raise ValueError(f'Unknown decoding: {self.decoding!r}')
        self.context_windows = settings['context_windows'] or 'disjoint'
        if self.context_windows not in CONTEXT_WINDOWS:
            raise ValueError(f'Unknown context windows: {self.context_windows!r}')

    def _windows(self, words):
        if self.context_windows == 'overlap':
            return encode_context_windows(self.tokenizer, words, max_length=self.max_length)
        return [(start, end, chunk, ids, range(start, end)) for start, end, chunk, ids
                in encode_chunks(self.tokenizer, words, max_length=self.max_length)]

    def predict_word_probabilities(self, words):
        import torch
        probabilities_by_word = [None] * len(words)
        with torch.inference_mode():
            for start, end, chunk, ids, owned in self._windows(words):
                tensors = {key: torch.tensor([value], device=self.device) for key, value in chunk.items()}
                probabilities = torch.stack([model(**tensors).logits[0].softmax(-1)
                                             for model in self.models]).mean(0)
                positions = word_label_positions(ids, chunk.tokens(), self.label_position)
                for word in owned:
                    probabilities_by_word[word] = [
                        float(value) for value in probabilities[positions[word - start]]]
        if any(row is None for row in probabilities_by_word):
            raise RuntimeError('Inference word coverage mismatch.')
        return probabilities_by_word

    def predict_words(self, words, repair=True, o_bias=None):
        probabilities = self.predict_word_probabilities(words)
        return decode_probabilities(
            probabilities, self.id2label,
            self.o_bias if o_bias is None else o_bias, repair=repair,
            label_biases=self.label_biases, decoding=self.decoding)

    def predict(self, text):
        words = segment_document(text)
        model_tags, scores = self.predict_words([w.text for w in words])
        tags = (prepare_display_tags([w.text for w in words], model_tags)
                if self.expand_display_boundaries else model_tags)
        tokens = []
        for word, tag, model_tag, score in zip(words, tags, model_tags, scores):
            token = {'text': word.text, 'start': word.start, 'end': word.end,
                     'tag': tag, 'score': score}
            if tag != model_tag:
                token['model_tag'] = model_tag
            tokens.append(token)
        return {'text': text, 'entities': collect_entities(
                    text, words, tags, scores, model_tags=model_tags),
                'tokens': tokens}


# Backward-compatible name used by the research repository.
KhmerNER = Neam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Local folder or Hugging Face model ID')
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    print(json.dumps(Neam(args.model).predict(args.text), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
