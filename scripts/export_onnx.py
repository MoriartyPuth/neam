"""Export an ensemble (or chosen members) to ONNX for the torch-free onnx backend.

    python scripts/export_onnx.py --source MoriartyPuth/neam --members m4-s46 --output neam-lite-onnx

Each member becomes members/<name>/model.onnx (fp32, opset 17, dynamic sequence
length) plus its config and tokenizer files. The PyTorch weights are not copied.
ensemble.json keeps the source's decoding and window settings. Every member is
checked against PyTorch on sample text before the folder is written; export fails
if any softmax probability differs by more than --tolerance.

Exporting needs torch; running the result needs only onnxruntime.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from neam.inference import ENSEMBLE_FILE, ONNX_FILE, resolve_model_path

COPY = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'spiece.model')
SAMPLE = ('លោក ហ៊ុន ម៉ាណែត បានជួបជាមួយក្រសួងសុខាភិបាលនៅរាជធានីភ្នំពេញ។ '
          'ក្រសួងអប់រំ យុវជន និងកីឡា បានប្រកាសលទ្ធផលថ្មី។ YOL 2026')


def export_member(source_dir, target_dir, opset):
    import numpy as np
    import onnxruntime
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(source_dir).eval()
    encoded = tokenizer(SAMPLE, return_tensors='pt')
    names = [n for n in ('input_ids', 'attention_mask', 'token_type_ids') if n in encoded]
    target_dir.mkdir(parents=True)
    path = target_dir / ONNX_FILE
    torch.onnx.export(
        model, tuple(encoded[n] for n in names), str(path),
        input_names=names, output_names=['logits'],
        dynamic_axes={n: {0: 'batch', 1: 'sequence'} for n in [*names, 'logits']},
        opset_version=opset, dynamo=False)
    for filename in COPY:
        if (Path(source_dir) / filename).exists():
            shutil.copy2(Path(source_dir) / filename, target_dir / filename)

    with torch.inference_mode():
        expected = model(**encoded).logits[0].softmax(-1).numpy()
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    logits = session.run(None, {n: encoded[n].numpy().astype(np.int64) for n in names})[0][0]
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    got = exp / exp.sum(-1, keepdims=True)
    return float(np.abs(got - expected).max()), bool((got.argmax(-1) == expected.argmax(-1)).all())


def export(source, output, names=None, revision=None, opset=17, tolerance=1e-4):
    root = Path(resolve_model_path(source, revision))
    spec = json.loads((root / ENSEMBLE_FILE).read_text(encoding='utf-8'))
    available = {Path(m).name: m for m in spec['members']}
    names = names or list(available)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f'Unknown members {missing}; available: {sorted(available)}')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new folder.')
    report = {}
    # Build in a temporary folder so a failed check leaves no partial output.
    with tempfile.TemporaryDirectory(dir=output.parent if output.parent.exists() else None) as tmp:
        staging = Path(tmp) / 'model'
        members = []
        for name in names:
            diff, same_argmax = export_member((root / available[name]).resolve(), staging / 'members' / name, opset)
            report[name] = {'max_probability_difference': diff, 'same_argmax': same_argmax}
            if diff > tolerance or not same_argmax:
                raise RuntimeError(f'{name}: ONNX differs from PyTorch (max diff {diff:.2e}, '
                                   f'same argmax: {same_argmax}).')
            members.append(f'members/{name}')
        settings = {k: v for k, v in spec.items() if k not in ('members', 'note')}
        (staging / ENSEMBLE_FILE).write_text(json.dumps(
            {'members': members, **settings,
             'note': f'ONNX export of {source}: {", ".join(names)}.'},
            indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        output.mkdir(parents=True, exist_ok=True)
        for item in staging.iterdir():
            shutil.move(str(item), output / item.name)
    return {'output': str(output), 'members': members, 'settings': settings, 'checks': report}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='MoriartyPuth/neam', help='Ensemble folder or Hub ID')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--members', nargs='*', default=None, help='Member names (default: all)')
    parser.add_argument('--output', required=True, help='New, empty output folder')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.members, args.revision, args.opset, args.tolerance),
                     indent=2))


if __name__ == '__main__':
    main()
