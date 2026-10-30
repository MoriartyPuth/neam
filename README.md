# Neam (នាម)

**Khmer named entity recognition for news.** Neam finds people (PER), organizations (ORG)
and places (LOC) in Khmer text and returns exact character offsets in your original input.
It runs offline on a CPU.

*នាម* means "name, noun": the *named* in named entity.

| | Exact-span F1 |
| --- | ---: |
| Held-out Khmer news (10 articles, 199 entities) | **86.7** |
| Validation Khmer news (5 articles, 111 entities) | 82.3 |

An entity counts only if both its boundaries and its type are exact. Retraining with
different seeds gives about 86–87 on average. See [MODEL_CARD.md](MODEL_CARD.md) for the
details and limitations.

## Install

```bash
pip install neam            # library
pip install "neam[app]"     # plus the web app
```

Python 3.10+. Install the PyTorch build that suits your machine first if you want GPU support.
The model (about 265 MB) downloads from the Hugging Face Hub on first use.

To work on the code itself, clone the repository and run `pip install -e ".[app]"`.

## Use

```python
from neam import Neam

ner = Neam.from_pretrained("MoriartyPuth/neam")   # or a local model folder
result = ner.predict("លោក ហ៊ុន ម៉ាណែត បានជួបជាមួយក្រសួងសុខាភិបាលនៅរាជធានីភ្នំពេញ។")
for entity in result["entities"]:
    print(entity["label"], entity["text"], entity["start"], entity["end"])
# PER លោក ហ៊ុន ម៉ាណែត 0 15
# ORG ក្រសួងសុខាភិបាល 27 42
# LOC រាជធានីភ្នំពេញ 44 58
```

From the command line:

```bash
python -m neam.inference --text "ក្រសួងអប់រំ យុវជន និងកីឡា បានប្រកាសលទ្ធផលថ្មី។"
```

Each entity has `label`, `text`, `start`, `end` (Python character offsets into the input) and
`score`, the model's mean confidence, which is not a calibrated probability.

## Web app

The web app is `app.py` in this repository. Run it from a clone after `pip install -e ".[app]"`:

```bash
python app.py                      # http://127.0.0.1:7860
python app.py --model /path/to/local/model
```

Paste Khmer text, or open **Import an online article** and give a public news URL. The page
is fetched with redirect and private-network checks, and the text is limited to 20,000
characters. Highlights: PER red, ORG green, LOC blue.

## How it works

1. **Segmentation.** Text is split into words with khmer-nltk. Only zero-width characters
   (U+200B/C/D) are dropped; offsets always point into the original string.
2. **Subword alignment.** Each word's label is read from its first subword that is not a bare
   SentencePiece `▁` marker.
3. **Overlapping windows.** Long texts are read in 128-subword windows that overlap. Each word
   is taken from the window where it has the most context on both sides.
4. **Ensemble and Viterbi.** Seven ALBERT models' probabilities are averaged and decoded into
   the best valid IOB2 sequence.
5. **Display rules.** Short honorifics (លោក) join a following name while ceremonial titles
   (ឯកឧត្តម, សម្តេច, …) stay outside it, place designators (ខេត្ត, ស្រុក, …) are completed,
   nationality phrases (ជនជាតិ + country) are not tagged as places, and punctuation,
   whitespace and zero-width characters are trimmed from span edges.

The labelling rules are in [docs/annotation_policy.md](docs/annotation_policy.md).

## Development

```bash
python -m unittest discover -s tests
```

`scripts/export_model.py` packages a trained ensemble folder into the Hub layout
(`ensemble.json` plus `members/`). Training code and research history are kept in a separate
research repository.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party attributions.
