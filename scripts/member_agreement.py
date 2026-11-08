"""Measure how closely each ensemble member agrees with the full ensemble.

This is the label-free selection rule behind Neam lite. It uses unlabelled text
only, so the gold validation and held-out sets stay untouched and the chosen
member's held-out score remains an unbiased estimate. Agreement is similarity to
the ensemble, not accuracy.

    python scripts/member_agreement.py --text-files a.txt b.txt --max-chars 3000

Every member is decoded exactly as in the full model (same Viterbi decoding,
overlapping windows, display rules and entity collection); only the averaged
members change.
"""
import argparse
import json
import time
from pathlib import Path

from neam.inference import (Neam, collect_entities, load_ensemble_spec, prepare_display_tags,
                            resolve_model_path)
from neam.preprocess import segment_document


def entity_set(ner, models, text):
    ner.models = models
    words = segment_document(text)
    tokens = [w.text for w in words]
    started = time.perf_counter()
    tags, scores = ner.predict_words(tokens)
    elapsed = time.perf_counter() - started
    display = prepare_display_tags(tokens, tags) if ner.expand_display_boundaries else tags
    entities = collect_entities(text, words, display, scores, model_tags=tags)
    return {(e['start'], e['end'], e['label']) for e in entities}, elapsed


def agreement(model, texts, revision=None):
    ner = Neam(model, revision=revision)
    members = [Path(m).name for m in load_ensemble_spec(resolve_model_path(model, revision))[0]]
    everything = list(ner.models)
    reference, ensemble_seconds = [], 0.0
    for text in texts:
        found, seconds = entity_set(ner, everything, text)
        reference.append(found)
        ensemble_seconds += seconds
    rows = []
    for name, member in zip(members, everything):
        tp = fp = fn = 0
        seconds = 0.0
        for text, ref in zip(texts, reference):
            found, elapsed = entity_set(ner, [member], text)
            seconds += elapsed
            tp, fp, fn = tp + len(found & ref), fp + len(found - ref), fn + len(ref - found)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({'member': name, 'f1': round(100 * f1, 1), 'precision': round(100 * precision, 1),
                     'recall': round(100 * recall, 1), 'seconds': round(seconds, 2)})
    ner.models = everything
    return {'ensemble_entities': sum(map(len, reference)), 'ensemble_seconds': round(ensemble_seconds, 2),
            'characters': sum(map(len, texts)), 'members': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='MoriartyPuth/neam')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--text-files', nargs='+', required=True, help='UTF-8 unlabelled Khmer text')
    parser.add_argument('--max-chars', type=int, default=3000, help='Characters read from each file')
    args = parser.parse_args()
    texts = [Path(p).read_text(encoding='utf-8')[:args.max_chars] for p in args.text_files]
    print(json.dumps(agreement(args.model, texts, args.revision), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
