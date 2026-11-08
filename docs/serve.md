# Neam server mode (stdio protocol 1)

`python -m neam.serve` runs Neam as a long-lived child process that another program
talks to over stdin/stdout. It suits desktop apps and other local tools: there is no
port, and the process ends when its parent closes stdin.

```bash
python -m neam.serve --model MoriartyPuth/neam            # Hub ID or local folder
python -m neam.serve --model ./neam-lite --device cpu --threads 2
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `MoriartyPuth/neam` | Local model folder or Hugging Face ID |
| `--revision` | latest | Hub revision |
| `--device` | `cuda` if available, else `cpu` | Torch device |
| `--threads` | torch default | CPU threads for inference |
| `--max-chars` | 20000 | Longest accepted `text` |
| `--backend` | `onnx` if the model has `model.onnx`, else `torch` | Inference backend (see below) |

## Transport

- **One JSON object per line**, UTF-8, `\n`-terminated, in both directions.
- stdout carries protocol messages only. Logs and warnings, including any output from
  dependencies, go to stderr.
- Requests are answered **in order**, one at a time. To abandon a running request,
  kill the process and start a new one.
- The process stops on EOF on stdin or after answering `shutdown`.
- The model starts loading in the background as soon as the process starts. `info` and
  `segment` answer immediately; `analyze` waits until loading finishes.

## Messages

```jsonc
{"id": 7, "method": "segment", "params": {"text": "..."}}             // request
{"id": 7, "result": {...}}                                              // success
{"id": 7, "error": {"code": "too_long", "message": "..."}}              // failure
```

`id` is a client-chosen integer, echoed back unchanged. If a request can't be read
(malformed JSON, a missing or non-integer `id`), the error response has `"id": null`.

### `info`

```json
{"neam": "0.3.0", "protocol": 1, "model": "MoriartyPuth/neam",
 "capabilities": ["segment", "analyze"], "offset_unit": "codepoint",
 "max_chars": 20000, "models_ready": false}
```

### `segment`

Wraps `neam.segment_document`. It doesn't need the model.

```json
{"words": [{"text": "ក្រសួង", "start": 27, "end": 33}]}
```

Punctuation comes back as separate words. Whitespace and zero-width characters
(U+200B/C/D) are never part of a word.

### `analyze`

Wraps `Neam.predict`.

```json
{"entities": [{"label": "ORG", "text": "ក្រសួងសុខាភិបាល", "start": 27, "end": 42, "score": 0.95}],
 "tokens":   [{"text": "ក្រសួង", "start": 27, "end": 33, "tag": "B-ORG", "score": 0.97}]}
```

Entities may carry `"boundary_adjusted": true`, and tokens may carry `model_tag`, when
the display rules changed the model's raw prediction. `score` is a confidence, not a
calibrated probability.

### `shutdown`

Returns `{}`; the process then exits with code 0.

## Offsets

`start`/`end` are **Unicode code point** indices into the request's `text`, half-open
(`[start, end)`). These match Python string indices, but not UTF-16 (JavaScript) or
UTF-8 byte offsets once the text contains characters outside the Basic Multilingual
Plane, such as emoji. Clients must convert.

## Errors

| Code | Meaning |
|---|---|
| `invalid_request` | Malformed JSON, a missing or non-integer `id`, or an unknown method |
| `invalid_input` | `text` is missing, not a string, contains unpaired surrogates, or was rejected by segmentation |
| `too_long` | `text` is over `max_chars`, or the request line is too long |
| `model_unavailable` | The model failed to load (the message says why); `segment` still works |
| `internal` | Unexpected failure; details are logged on stderr |

## Compatibility

`protocol` is incremented only for incompatible changes. New methods, new result
fields and new `info` fields are compatible additions; clients should ignore fields
they don't recognise.

## Neam lite

`scripts/subset_model.py` builds a model folder from chosen ensemble members, keeping
the ensemble's decoding and window settings:

```bash
python scripts/subset_model.py --members m4-s46 --output neam-lite
python -m neam.serve --model ./neam-lite
```

`m4-s46` is the member that agrees most closely with the full ensemble on unlabelled
text (`scripts/member_agreement.py`). It is about 37 MB and needs about 1/7 of the
ensemble's compute. Its accuracy on the gold sets has not yet been measured or published.

## Running without PyTorch (ONNX)

`scripts/export_onnx.py` converts members to ONNX. It checks every member against
PyTorch and refuses to write the folder if any probability differs by more than 1e-4:

```bash
python scripts/export_onnx.py --members m4-s46 --output neam-lite-onnx   # needs torch
```

A folder whose members contain `model.onnx` runs on ONNX Runtime automatically, and
then needs no PyTorch at all. A plain `pip install neam` is enough:

```bash
pip install neam
python -m neam.serve --model ./neam-lite-onnx
```

For apps that bundle Neam, `requirements-slim.txt` also leaves out SciPy and
scikit-learn, which khmer-nltk declares but its segmenter does not use. Segmentation
was verified identical on 46k characters:

```bash
pip install -r requirements-slim.txt
pip install --no-deps khmer-nltk sklearn-crfsuite neam
```

Measured on a Ryzen 7 6800H with 4 threads, on about 9,000 characters of Khmer
Wikipedia and news text:

| | PyTorch | ONNX |
|---|---:|---:|
| Entities and per-word tags | — | **identical** (lite: 118 entities, full: 115) |
| Lite (1 member), 9k characters | 4.5 s | 2.7 s |
| Full ensemble (7 members), 9k characters | 29.9 s | 19.0 s |
| Server peak memory (lite) | 534 MB | 319 MB (228 MB slim) |
| Server start → analyze → exit (lite) | 5.1 s | 4.2 s (1.3 s slim) |
| Python environment | 1.3 GB | 381 MB (248 MB slim) |

The ONNX backend runs on the CPU only.
