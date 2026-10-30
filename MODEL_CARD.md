---
language:
- km
license: apache-2.0
library_name: transformers
pipeline_tag: token-classification
base_model: seanghay/albert-khmer-small
datasets:
- unimelb-nlp/wikiann
- whoisjones/fiNERweb
tags:
- token-classification
- named-entity-recognition
- khmer
- ner
metrics:
- f1
---

# Neam (នាម): Khmer news NER

Neam finds **people (PER), organizations (ORG) and places (LOC)** in Khmer text and returns
character offsets in the original input. It was trained and evaluated on Khmer news.

> **Use the `neam` package to run this model.** It is an ensemble of seven ALBERT
> checkpoints (`members/`), combined by `ensemble.json` with Viterbi decoding, overlapping
> context windows and Khmer-specific display rules. Loading one member with the standard
> `transformers` pipeline gives noticeably worse results.

```bash
pip install neam
```

```python
from neam import Neam

ner = Neam.from_pretrained("MoriartyPuth/neam")
result = ner.predict("លោក ហ៊ុន ម៉ាណែត បានជួបជាមួយក្រសួងសុខាភិបាលនៅរាជធានីភ្នំពេញ។")
for entity in result["entities"]:
    print(entity["label"], entity["text"], entity["start"], entity["end"])
```

## Model details

- **Architecture:** seven fine-tuned `seanghay/albert-khmer-small` token classifiers (seeds 42
  to 48), with softmax probabilities averaged. Labels are IOB2 over khmer-nltk words, read from
  each word's first non-`▁` subword.
- **Decoding:** constrained IOB2 Viterbi with calibrated label scores (O × 0.9, ORG × 1.2,
  LOC × 0.85). Each word is read in the overlapping 128-subword window that gives it the most
  context.
- **Display rules:**
  - A short honorific such as លោក is joined to a following name; ceremonial titles and ranks
    (ឯកឧត្តម, សម្តេច, …) are kept outside it.
  - Common place designators (ខេត្ត, ស្រុក, …) are completed.
  - A LOC directly after ជនជាតិ (a nationality phrase) is removed.
  - Sentence punctuation, whitespace and zero-width characters are trimmed from span edges.
- **Size:** about 37 MB per member, about 265 MB in total. It runs on a CPU.

## Intended use and annotation policy

Neam is for tagging named entities in Khmer **news** text, for search, indexing or reading
aids. It follows `docs/annotation_policy.md`, in brief:

- PER covers names. Ceremonial titles (ឯកឧត្តម, សម្តេច, លោកជំទាវ) and ranks are excluded;
  short honorifics (លោក, លោកស្រី) are included.
- ORG covers named institutions, including their designator and any country or province in
  the name (for example រាជរដ្ឋាភិបាលកម្ពុជា and អយ្យការអមសាលាដំបូងខេត្តព្រះសីហនុ).
- LOC covers places and facilities.
- Nationality phrases (ជនជាតិ + country), products, events and storms are not entities.

## Training data

- Cleaned Khmer WikiANN and a 1,200-record Khmer fiNERweb sample (PER/ORG/LOC labels only).
- 117 records from the project's earlier Khmer news sets, plus 31 recent articles from Fresh
  News, Khmer Press and Kampuchea Thmey (821 entities), chosen where earlier ensembles
  disagreed most and labelled at character level under the policy. The labels are assistant
  drafts following the project owner's policy decisions; the owner, a Khmer reader, reviewed
  the newest 10 articles (399 labels, one correction) and a sample of five earlier articles
  (121 labels, no errors).
- A small synthetic curriculum contrasting nationality phrases with the same country used as
  a place.

Publisher article text is not redistributed with this model.

Training settings: 5 epochs, learning rate 3e-5, batch size 8 with 2-step gradient
accumulation, frozen embeddings, and the news data weighted ×3.

## Evaluation

Exact character-span F1: an entity counts only if both its boundaries and its type match.

| Set | Articles / entities | F1 | Precision | Recall | PER | ORG | LOC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 5 / 111 | 82.25 | 79.2 | 85.6 | 100.0 | 79.1 | 84.8 |
| Held-out (never used for training) | 10 / 199 | **86.70** | 85.0 | 88.4 | 98.5 | 74.5 | 91.8 |

The PER, ORG and LOC columns are F1. For comparison, the project's earliest model
(WikiANN-centred, no news-specific methods) scores 59.5 F1 on the same held-out set.

The validation labels follow the current policy (four bracketed-acronym and title labels were
corrected on 2026-10-30, 108 → 111 entities). The previous release (0.1.0, three seeds of the
v24 recipe) scores 79.65 / 85.71 on these labels.

**Seed variance.** Single models of this recipe score between 80.5 and 88.6 on held-out,
depending on the random seed; five-model ensembles average 86.8 (range 84.8–88.9). This
release uses all seven trained seeds, with no subset chosen after scoring. **Expect about
86–87 F1** on similar Khmer news. The newest training articles come from the same sources
and weeks as the held-out set, which may flatter the held-out score slightly.

## Limitations

- **Domain.** Evaluated only on news from three Cambodian publishers, collected in
  September–October 2026. Accuracy on social media, government documents, informal or
  older text is untested and likely lower.
- **Organization boundaries are the main weakness** (about 75 ORG F1 on held-out). Long
  institutional names are sometimes cut short or split into designator plus name.
- **Small evaluation sets.** 307 entities in total, labelled by the same process as the
  training data. There has been no independent second-annotator agreement study.
- **Segmentation.** It relies on khmer-nltk word segmentation. Unusual spellings and
  transliterated foreign names can be segmented in ways that hurt boundaries.
- Scores in the output are model confidences, not calibrated probabilities of correctness.

## License and attribution

- This model and the `neam` code are released under **Apache-2.0**.
- Base model: [`seanghay/albert-khmer-small`](https://huggingface.co/seanghay/albert-khmer-small)
  (Apache-2.0).
- Training data:
  - [`whoisjones/fiNERweb`](https://huggingface.co/datasets/whoisjones/fiNERweb) (MIT).
  - [`unimelb-nlp/wikiann`](https://huggingface.co/datasets/unimelb-nlp/wikiann). Its license is
    listed as unknown on the Hub; it is derived from Wikipedia.
  - News articles remain the property of their publishers and are not redistributed.
