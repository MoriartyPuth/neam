import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from neam.preprocess import Word
from neam.serve import PROTOCOL, Server, serve

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        return {'text': text,
                'entities': [{'start': 0, 'end': len(text), 'label': 'LOC', 'text': text, 'score': 0.9}],
                'tokens': [{'text': text, 'start': 0, 'end': len(text), 'tag': 'B-LOC', 'score': 0.9}]}


def fake_segment(text):
    if 'reject' in text:
        raise ValueError('Segmentation changed or skipped non-whitespace text.')
    return [Word(part, text.index(part), text.index(part) + len(part)) for part in text.split()]


def run_lines(server, *requests):
    stdin = io.BytesIO(b''.join(
        (r if isinstance(r, bytes) else json.dumps(r, ensure_ascii=False).encode('utf-8')) + b'\n'
        for r in requests))
    stdout = io.StringIO()
    serve(server, stdin, stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


class ServerTests(unittest.TestCase):
    def make(self, load=None, **kwargs):
        return Server(load or FakeModel, 'fake/model', segment=fake_segment, **kwargs)

    def test_info_before_and_after_loading(self):
        server = self.make()
        info = server.handle({'id': 1, 'method': 'info'})['result']
        self.assertEqual(info['protocol'], PROTOCOL)
        self.assertEqual(info['offset_unit'], 'codepoint')
        self.assertEqual(info['capabilities'], ['segment', 'analyze'])
        self.assertFalse(info['models_ready'])
        server.handle({'id': 2, 'method': 'analyze', 'params': {'text': 'x'}})
        self.assertTrue(server.handle({'id': 3, 'method': 'info'})['result']['models_ready'])

    def test_segment_does_not_need_the_model(self):
        def broken():
            raise RuntimeError('no weights')
        server = self.make(load=broken)
        response = server.handle({'id': 7, 'method': 'segment', 'params': {'text': 'ក ខ'}})
        self.assertEqual(response, {'id': 7, 'result': {'words': [
            {'text': 'ក', 'start': 0, 'end': 1}, {'text': 'ខ', 'start': 2, 'end': 3}]}})
        failed = server.handle({'id': 8, 'method': 'analyze', 'params': {'text': 'ក'}})
        self.assertEqual(failed['error']['code'], 'model_unavailable')
        self.assertIn('no weights', failed['error']['message'])

    def test_analyze_returns_entities_and_tokens_only(self):
        result = self.make().handle({'id': 1, 'method': 'analyze', 'params': {'text': 'ភ្នំពេញ'}})['result']
        self.assertEqual(set(result), {'entities', 'tokens'})
        self.assertEqual(result['entities'][0]['label'], 'LOC')

    def test_errors(self):
        server = self.make(max_chars=5)
        cases = [
            ({'id': 1, 'method': 'segment', 'params': {'text': '123456'}}, 'too_long'),
            ({'id': 1, 'method': 'segment', 'params': {}}, 'invalid_input'),
            ({'id': 1, 'method': 'segment', 'params': {'text': 5}}, 'invalid_input'),
            ({'id': 1, 'method': 'segment', 'params': {'text': 'a\ud800'}}, 'invalid_input'),
            ({'id': 1, 'method': 'segment', 'params': {'text': 'reject'}}, 'too_long'),
            ({'id': 1, 'method': 'nope'}, 'invalid_request'),
            ({'id': '1', 'method': 'info'}, 'invalid_request'),
            ({'id': True, 'method': 'info'}, 'invalid_request'),
            (['not', 'an', 'object'], 'invalid_request'),
        ]
        for request, code in cases:
            with self.subTest(request=request):
                self.assertEqual(server.handle(request)['error']['code'], code)
        self.assertEqual(self.make().handle(
            {'id': 2, 'method': 'segment', 'params': {'text': 'reject'}})['error']['code'], 'invalid_input')

    def test_internal_errors_are_reported_not_raised(self):
        class Exploding:
            def predict(self, text):
                raise KeyError('boom')
        response = self.make(load=Exploding).handle({'id': 4, 'method': 'analyze', 'params': {'text': 'x'}})
        self.assertEqual(response['id'], 4)
        self.assertEqual(response['error']['code'], 'internal')

    def test_stream_framing(self):
        server = self.make(max_chars=10)
        responses = run_lines(
            server,
            {'id': 1, 'method': 'info'},
            b'',
            b'{not json',
            b'\xff\xfe',
            b'{"id": 2, "method": "segment", "params": {"text": "' + b'x' * 200_000 + b'"}}',
            {'id': 3, 'method': 'segment', 'params': {'text': 'ok'}},
            {'id': 4, 'method': 'shutdown'},
            {'id': 5, 'method': 'info'},  # never answered: the server stopped
        )
        self.assertEqual([r.get('id') for r in responses], [1, None, None, None, 3, 4])
        self.assertEqual([r['error']['code'] for r in responses[1:4]], ['invalid_request', 'invalid_request', 'too_long'])
        self.assertEqual(responses[4]['result']['words'][0]['text'], 'ok')
        self.assertTrue(server.stopped)


class SubprocessTests(unittest.TestCase):
    def test_stdout_carries_only_protocol_lines(self):
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': str(ROOT)}
        requests = [
            {'id': 1, 'method': 'info'},
            {'id': 2, 'method': 'segment', 'params': {'text': 'ខ្ញុំ😀ចូលចិត្ត​អាន។'}},
            {'id': 3, 'method': 'analyze', 'params': {'text': 'ភ្នំពេញ'}},
            {'id': 4, 'method': 'shutdown'},
        ]
        stdin = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests).encode('utf-8')
        proc = subprocess.run(
            [sys.executable, '-m', 'neam.serve', '--model', str(ROOT / 'tests' / 'no-such-model')],
            input=stdin, capture_output=True, env=env, timeout=120, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode('utf-8', 'replace'))
        lines = proc.stdout.decode('utf-8').splitlines()
        responses = [json.loads(line) for line in lines]  # every stdout line is protocol JSON
        self.assertEqual([r['id'] for r in responses], [1, 2, 3, 4])
        words = responses[1]['result']['words']
        text = requests[1]['params']['text']
        for w in words:  # code point offsets into the original text, ZWSP skipped
            self.assertEqual(text[w['start']:w['end']], w['text'])
        self.assertIn('😀', [w['text'] for w in words])
        self.assertEqual(responses[2]['error']['code'], 'model_unavailable')
        self.assertIn(b'khmer-nltk', proc.stderr)  # dependency logging stayed on stderr


if __name__ == '__main__':
    unittest.main()
