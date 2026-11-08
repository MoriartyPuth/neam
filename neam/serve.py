"""Neam over stdio: one JSON object per line in each direction (protocol 1).

Run ``python -m neam.serve --model <folder or Hub ID>``. The protocol is described in
docs/serve.md. stdout carries protocol messages only; everything else (logs,
warnings, stray prints from dependencies) is redirected to stderr.
"""
import argparse
import json
import logging
import os
import sys
import threading
import traceback

from neam import __version__
from neam.inference import DEFAULT_MODEL, Neam
from neam.preprocess import segment_document

PROTOCOL = 1
DEFAULT_MAX_CHARS = 20_000
CAPABILITIES = ('segment', 'analyze')
# A JSON-escaped character takes at most 12 bytes (a \uXXXX surrogate pair);
# the rest of a request is small. Longer lines are rejected without parsing.
_LINE_OVERHEAD = 4096

log = logging.getLogger('neam.serve')


class ProtocolError(Exception):
    """An error reported to the client as {"error": {"code", "message"}}."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(request_id, code, message):
    return {'id': request_id, 'error': {'code': code, 'message': message}}


class Server:
    """Handles decoded requests. Transport-free so it can be tested directly.

    ``load`` builds the model; it runs once, in a background thread, after
    ``start_loading``. ``segment`` works without the model.
    """

    def __init__(self, load, model_name, max_chars=DEFAULT_MAX_CHARS, segment=segment_document):
        self._load = load
        self._segment = segment
        self.model_name = model_name
        self.max_chars = max_chars
        self.stopped = False
        self._model = None
        self._load_error = None
        self._ready = threading.Event()
        self._loader = None

    def start_loading(self):
        if self._loader is None:
            self._loader = threading.Thread(target=self._load_model, name='neam-load', daemon=True)
            self._loader.start()

    def _load_model(self):
        try:
            self._model = self._load()
            log.info('model loaded: %s', self.model_name)
        except Exception as exc:  # reported to clients as model_unavailable
            self._load_error = f'{type(exc).__name__}: {exc}'
            log.error('model failed to load: %s', self._load_error)
        finally:
            self._ready.set()

    def _wait_for_model(self):
        self.start_loading()
        self._ready.wait()
        if self._model is None:
            raise ProtocolError('model_unavailable', self._load_error or 'The model could not be loaded.')
        return self._model

    def _text_param(self, params):
        if not isinstance(params, dict) or not isinstance(params.get('text'), str):
            raise ProtocolError('invalid_input', 'params.text must be a string.')
        text = params['text']
        if len(text) > self.max_chars:
            raise ProtocolError('too_long', f'text has {len(text)} characters; the limit is {self.max_chars}.')
        if any('\ud800' <= c <= '\udfff' for c in text):
            raise ProtocolError('invalid_input', 'text contains unpaired surrogate code points.')
        return text

    def handle(self, request):
        """Return the response object for one decoded request."""
        if not isinstance(request, dict):
            return _error(None, 'invalid_request', 'A request must be a JSON object.')
        request_id = request.get('id')
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            return _error(None, 'invalid_request', 'id must be an integer.')
        method = request.get('method')
        try:
            if method == 'info':
                result = self.info()
            elif method == 'segment':
                text = self._text_param(request.get('params'))
                try:
                    words = self._segment(text)
                except ValueError as exc:
                    raise ProtocolError('invalid_input', str(exc)) from exc
                result = {'words': [{'text': w.text, 'start': w.start, 'end': w.end} for w in words]}
            elif method == 'analyze':
                text = self._text_param(request.get('params'))
                model = self._wait_for_model()
                try:
                    prediction = model.predict(text)
                except ValueError as exc:
                    raise ProtocolError('invalid_input', str(exc)) from exc
                result = {'entities': prediction['entities'], 'tokens': prediction['tokens']}
            elif method == 'shutdown':
                self.stopped = True
                result = {}
            else:
                raise ProtocolError('invalid_request', f'Unknown method: {method!r}.')
        except ProtocolError as exc:
            return _error(request_id, exc.code, exc.message)
        except Exception as exc:
            log.error('request %s (%s) failed:\n%s', request_id, method, traceback.format_exc())
            return _error(request_id, 'internal', f'{type(exc).__name__}: {exc}'[:500])
        return {'id': request_id, 'result': result}

    def info(self):
        return {
            'neam': __version__,
            'protocol': PROTOCOL,
            'model': self.model_name,
            'capabilities': list(CAPABILITIES),
            'offset_unit': 'codepoint',
            'max_chars': self.max_chars,
            'models_ready': self._ready.is_set() and self._model is not None,
        }


def serve(server, stdin, stdout):
    """Read requests from binary ``stdin`` and write responses to text ``stdout``."""
    limit = server.max_chars * 12 + _LINE_OVERHEAD

    def send(response):
        stdout.write(json.dumps(response, ensure_ascii=False, separators=(',', ':')) + '\n')
        stdout.flush()

    while not server.stopped:
        raw = stdin.readline(limit + 1)
        if not raw:
            break  # EOF: the client went away
        if len(raw) > limit and not raw.endswith(b'\n'):
            while True:  # discard the rest of the oversized line
                rest = stdin.readline(65536)
                if not rest or rest.endswith(b'\n'):
                    break
            send(_error(None, 'too_long', 'Request line is too long.'))
            continue
        if not raw.strip():
            continue
        try:
            request = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            send(_error(None, 'invalid_request', f'Malformed request: {exc}'))
            continue
        send(server.handle(request))


def _protocol_stdout():
    """Keep the real stdout for protocol messages; send everything else to stderr."""
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', newline='\n')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Local model folder or Hugging Face model ID')
    parser.add_argument('--revision', default=None, help='Hub revision (ignored for local folders)')
    parser.add_argument('--device', default=None, help='Torch device, e.g. cpu (default: cuda if available)')
    parser.add_argument('--threads', type=int, default=None, help='CPU threads for inference')
    parser.add_argument('--max-chars', type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args(argv)
    if args.max_chars < 1:
        parser.error('--max-chars must be positive')

    protocol_out = _protocol_stdout()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='neam.serve %(levelname)s %(message)s')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    def load():
        if args.threads:
            import torch
            torch.set_num_threads(args.threads)
        return Neam(args.model, device=args.device, revision=args.revision)

    name = f'{args.model}@{args.revision}' if args.revision else args.model
    server = Server(load, name, max_chars=args.max_chars)
    server.start_loading()
    try:
        serve(server, sys.stdin.buffer, protocol_out)
    except KeyboardInterrupt:
        pass
    finally:
        protocol_out.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
