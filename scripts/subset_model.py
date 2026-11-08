"""Build a self-contained model folder from a subset of an ensemble's members.

Used for the single-member "Neam lite" release. Decoding and windowing settings are
copied from the source ensemble.json. Without them a lone member falls back to argmax
decoding and disjoint windows, and gives worse results.

    python scripts/subset_model.py --source MoriartyPuth/neam --members m4-s46 --output out/neam-lite

The output holds ensemble.json plus members/<name>/ with only the files inference
needs. Nothing is uploaded.
"""
import argparse
import json
import shutil
from pathlib import Path

from neam.inference import ENSEMBLE_FILE, resolve_model_path

KEEP = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'spiece.model')


def subset(source, names, output, note=None, revision=None):
    root = Path(resolve_model_path(source, revision))
    spec = json.loads((root / ENSEMBLE_FILE).read_text(encoding='utf-8'))
    available = {Path(member).name: member for member in spec['members']}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f'Unknown members {missing}; available: {sorted(available)}')
    if len(set(names)) != len(names):
        raise ValueError('Members must be distinct.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new folder.')
    members = []
    for name in names:
        source_dir = (root / available[name]).resolve()
        target = output / 'members' / name
        target.mkdir(parents=True)
        for filename in KEEP:
            if (source_dir / filename).exists():
                shutil.copy2(source_dir / filename, target / filename)
        if not (target / 'model.safetensors').exists():
            raise ValueError(f'{source_dir} has no model.safetensors.')
        members.append(f'members/{name}')
    settings = {key: value for key, value in spec.items() if key not in ('members', 'note')}
    manifest = {'members': members, **settings,
                'note': note or f'Subset of {source}: {", ".join(names)}.'}
    (output / ENSEMBLE_FILE).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return {'output': str(output), 'members': members, 'settings': settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='MoriartyPuth/neam', help='Ensemble folder or Hub ID')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--members', nargs='+', required=True, help='Member folder names, e.g. m4-s46')
    parser.add_argument('--output', required=True, help='New, empty output folder')
    parser.add_argument('--note', default=None)
    args = parser.parse_args()
    print(json.dumps(subset(args.source, args.members, args.output, args.note, args.revision), indent=2))


if __name__ == '__main__':
    main()
