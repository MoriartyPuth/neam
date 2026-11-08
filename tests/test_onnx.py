import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_TORCH = importlib.util.find_spec('torch') is not None
HAVE_ORT = importlib.util.find_spec('onnxruntime') is not None


def load_export_script():
    spec = importlib.util.spec_from_file_location('export_onnx', ROOT / 'scripts' / 'export_onnx.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_tiny_model(directory):
    """A tiny trained-shape BERT token classifier with the Neam labels."""
    from transformers import BertConfig, BertForTokenClassification
    from neam.preprocess import LABEL2ID, LABELS
    try:
        from tests.test_neam import tiny_tokenizer
    except ImportError:  # run as `unittest discover -s tests`
        from test_neam import tiny_tokenizer
    import torch
    torch.manual_seed(0)
    model = BertForTokenClassification(BertConfig(
        vocab_size=7, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=32, num_labels=7,
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID))
    model.config.khmer_ner_max_length = 8
    model.config.khmer_ner_label_position = 'first-content-subword'
    model.save_pretrained(directory)
    tiny_tokenizer().save_pretrained(directory)


@unittest.skipUnless(HAVE_TORCH and HAVE_ORT, 'needs torch and onnxruntime')
class OnnxBackendTests(unittest.TestCase):
    def test_export_and_predictions_match_torch(self):
        from neam.inference import Neam
        export = load_export_script()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp, 'source', 'members', 'm0')
            source.mkdir(parents=True)
            save_tiny_model(source)
            Path(tmp, 'source', 'ensemble.json').write_text(json.dumps(
                {'members': ['members/m0', 'members/m0'], 'decoding': 'viterbi', 'context_windows': 'overlap'}))
            out = Path(tmp, 'onnx')
            result = export.export(str(Path(tmp, 'source')), out, names=['m0'])
            self.assertTrue(result['checks']['m0']['same_argmax'])
            self.assertFalse((out / 'members' / 'm0' / 'model.safetensors').exists())
            spec = json.loads((out / 'ensemble.json').read_text())
            self.assertEqual((spec['decoding'], spec['context_windows']), ('viterbi', 'overlap'))

            torch_ner = Neam(str(Path(tmp, 'source')), device='cpu', backend='torch')
            onnx_ner = Neam(str(out))  # auto-detects model.onnx
            self.assertEqual(onnx_ner.backend, 'onnx')
            self.assertEqual((onnx_ner.decoding, onnx_ner.context_windows), ('viterbi', 'overlap'))
            words = ['ab', 'c'] * 9  # several overlapping windows at max_length 8
            for a, b in zip(torch_ner.predict_word_probabilities(words),
                            onnx_ner.predict_word_probabilities(words)):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=5)
            self.assertEqual(torch_ner.predict_words(words)[0], onnx_ner.predict_words(words)[0])

            # Two ONNX members average like two torch members.
            spec['members'] = ['members/m0', 'members/m0']
            (out / 'ensemble.json').write_text(json.dumps(spec))
            pair = Neam(str(out))
            self.assertEqual(len(pair.models), 2)
            for a, b in zip(onnx_ner.predict_word_probabilities(words), pair.predict_word_probabilities(words)):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=6)

    def test_backend_selection_and_errors(self):
        from neam.inference import Neam, detect_backend
        with tempfile.TemporaryDirectory() as tmp:
            save_tiny_model(tmp)
            self.assertEqual(detect_backend([tmp]), 'torch')
            with self.assertRaises(FileNotFoundError):
                Neam(tmp, backend='onnx')
            with self.assertRaises(ValueError):
                Neam(tmp, backend='tensorflow')
            Path(tmp, 'model.onnx').write_bytes(b'')
            self.assertEqual(detect_backend([tmp]), 'onnx')
            with self.assertRaises(ValueError):
                Neam(tmp, backend='onnx', device='cuda')


if __name__ == '__main__':
    unittest.main()
