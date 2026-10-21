"""Neam web app: paste Khmer text or import a public news article URL."""
import argparse
from functools import lru_cache

import gradio as gr
from neam.articles import MAX_INPUT_CHARS, extract_article_text
from neam.inference import DEFAULT_MODEL, Neam

EXAMPLES = [
    ['ហ៊ុន សែន បានទៅប្រទេសចិន។'],
    ['ក្រសួងអប់រំ យុវជន និងកីឡា បានប្រកាសលទ្ធផលថ្មី។'],
    ['សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញស្ថិតនៅរាជធានីភ្នំពេញ។'],
    ['សុខា និង ដារ៉ា រស់នៅរាជធានីភ្នំពេញ។'],
    ['ថ្ងៃនេះអាកាសធាតុល្អណាស់។'],
]


def create_app(model_path):
    @lru_cache(maxsize=1)
    def model():
        return Neam(model_path)

    def predict(text):
        if len(text) > MAX_INPUT_CHARS:
            raise gr.Error(f'Please limit input to {MAX_INPUT_CHARS:,} characters.')
        if not text.strip():
            return {'text': text, 'entities': []}, []
        try:
            result = model().predict(text)
        except (ValueError, OSError) as exc:
            raise gr.Error(f'Prediction unavailable: {exc}') from exc
        highlights = {'text': text, 'entities': [
            {'entity': e['label'], 'start': e['start'], 'end': e['end']} for e in result['entities']]}
        return highlights, result['entities']

    def load_article(url):
        try:
            return extract_article_text(url)
        except (ValueError, OSError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title='Neam: Khmer Named Entity Recognition') as demo:
        gr.Markdown('# Neam (នាម)\nKhmer named entity recognition: people (PER), organizations (ORG), and places (LOC). Trained and evaluated on Khmer news.')
        with gr.Accordion('Import an online article', open=False):
            url = gr.Textbox(label='Public article URL', placeholder='https://example.com/article')
            article_button = gr.Button('Load article text')
            import_status = gr.Markdown()
        text = gr.Textbox(label='Khmer text', lines=6, placeholder='បញ្ចូលអត្ថបទខ្មែរនៅទីនេះ…')
        button = gr.Button('Identify entities', variant='primary')
        highlighted = gr.HighlightedText(label='Entities', combine_adjacent=False,
                                          color_map={'PER': '#ef4444', 'LOC': '#3b82f6', 'ORG': '#22c55e'})
        details = gr.JSON(label='Entity offsets and model scores')
        gr.Examples(examples=EXAMPLES, inputs=text, label='Try an example')
        gr.Markdown('Scores reflect model confidence and are not calibrated probabilities of correctness.')
        button.click(predict, inputs=text, outputs=[highlighted, details], concurrency_limit=1)
        text.submit(predict, inputs=text, outputs=[highlighted, details], concurrency_limit=1)
        article_button.click(load_article, inputs=url, outputs=[text, import_status], concurrency_limit=1)
    return demo.queue(max_size=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Local folder or Hugging Face model ID')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    create_app(args.model).launch(server_name=args.host, server_port=args.port, share=False)
