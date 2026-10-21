"""Package a research ensemble folder as a self-contained Hugging Face model folder.

The source is an ensemble folder whose ensemble.json lists member checkpoints by
relative path (for example the research repository's
models/khmer-ner-news-v24-ensemble). The output holds ensemble.json, MODEL_CARD.md
as README.md, and members/<name>/ with only the files inference needs. Upload the
output folder to the Hub yourself; this script never publishes anything.
"""
import argparse
import json
import shutil
from pathlib import Path

KEEP = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'spiece.model')


def export(source, output, card=None):
    source, output = Path(source), Path(output)
    spec = json.loads((source / 'ensemble.json').read_text(encoding='utf-8'))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty; choose a new folder.')
    members = []
    for index, member in enumerate(spec['members']):
        member_path = (source / member).resolve()
        name = f'm{index}-{member_path.name}'
        target = output / 'members' / name
        target.mkdir(parents=True)
        for filename in KEEP:
            if (member_path / filename).exists():
                shutil.copy2(member_path / filename, target / filename)
        if not (target / 'model.safetensors').exists():
            raise ValueError(f'{member_path} has no model.safetensors.')
        members.append(f'members/{name}')
    settings = {key: value for key, value in spec.items() if key not in ('members', 'note')}
    (output / 'ensemble.json').write_text(json.dumps(
        {'members': members, **settings,
         'note': 'Neam ensemble: member softmax probabilities are averaged, then decoded.'},
        indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if card:
        shutil.copy2(card, output / 'README.md')
    return {'output': str(output), 'members': members, 'settings': settings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Research ensemble folder with ensemble.json')
    parser.add_argument('--output', required=True, help='New, empty output folder')
    parser.add_argument('--card', default=str(Path(__file__).resolve().parents[1] / 'MODEL_CARD.md'))
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output, args.card), indent=2))


if __name__ == '__main__':
    main()
