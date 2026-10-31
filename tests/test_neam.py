"""Inference, alignment and article-import tests for Neam (no model download needed)."""
import json
import tempfile
import unittest
from pathlib import Path

from neam.inference import (collect_entities, expand_entity_boundaries,
                            prepare_display_tags, repair_iob2)
from neam.preprocess import (LABELS, Word, align_labels, encode_chunks, normalize, segment,
                             segment_document, spans_to_words)


class PreprocessingTests(unittest.TestCase):
    def test_article_extraction_and_private_url_rejection(self):
        from neam.articles import _validate_public_url, article_text_from_html
        html = '<nav>Menu</nav><article><h1>ព័ត៌មាន</h1><p>លោក សុខា នៅភ្នំពេញ។</p></article><script>x()</script>'
        self.assertEqual(article_text_from_html(html), 'ព័ត៌មាន\nលោក សុខា នៅភ្នំពេញ។')
        article = '<div class="content"><p>' + ('អត្ថបទព័ត៌មាន ' * 20) + '</p></div>'
        related = '<main><h2>ព័ត៌មានផ្សេងទៀត</h2>' + article + '<p>លោក មិនពាក់ព័ន្ធ</p></main>'
        self.assertNotIn('មិនពាក់ព័ន្ធ', article_text_from_html(related))
        wrapped = '<article><h1>ចំណងជើង</h1><div class="td-post-content"><p>' \
                  + '<span>Share</span>' + ('ខ្លឹមសារព័ត៌មាន ' * 20) \
                  + '</p></div><p>អត្ថបទពាក់ព័ន្ធ</p></article>'
        extracted = article_text_from_html(wrapped)
        self.assertNotIn('អត្ថបទពាក់ព័ន្ធ', extracted)
        self.assertNotIn('Share', extracted)
        thmey = '<div class="post_details_block"><div class="adaptive"><p>' \
                + ('ខ្លឹមសារអត្ថបទ ' * 20) \
                + '</p></div></div><div class="main-content"><p>' \
                + ('ព័ត៌មានពេញនិយម ' * 20) + '</p></div>'
        extracted = article_text_from_html(thmey)
        self.assertIn('ខ្លឹមសារអត្ថបទ', extracted)
        self.assertNotIn('ព័ត៌មានពេញនិយម', extracted)
        kampuchea = '<div id="mvp-content-main">' + ('អត្ថបទកម្ពុជា ' * 20) \
                    + '</div><div class="mvp-related-posts">ព័ត៌មានពាក់ព័ន្ធ</div>'
        extracted = article_text_from_html(kampuchea)
        self.assertIn('អត្ថបទកម្ពុជា', extracted)
        self.assertNotIn('ព័ត៌មានពាក់ព័ន្ធ', extracted)
        # English menus and footers around a Khmer article are dropped; inner lines stay.
        framed = '<main><p>Home</p><p>News</p><p>ចំណងជើង</p><p>31-Oct-2026</p>' \
                 + '<p>Ferdinand Marcos</p><p>' + ('ខ្លឹមសារ ' * 30) + '</p>' \
                 + '<p>Ministry of Interior</p><p>All rights reserved</p></main>'
        self.assertEqual(article_text_from_html(framed).splitlines(),
                         ['ចំណងជើង', '31-Oct-2026', 'Ferdinand Marcos', ('ខ្លឹមសារ ' * 30).strip()])
        english = '<main><p>Only English text here.</p></main>'
        self.assertEqual(article_text_from_html(english), 'Only English text here.')
        with self.assertRaisesRegex(ValueError, 'public internet'):
            _validate_public_url('http://127.0.0.1/private')

    def test_multiline_inference_segmentation_offsets(self):
        words = segment_document('សុខា\n\nភ្នំពេញ', lambda value: [value] if value else [])
        self.assertEqual(words, [Word('សុខា', 0, 4), Word('ភ្នំពេញ', 6, 13)])

    def test_khmer_display_boundary_expansion(self):
        words = ['ក្រសួង', 'ការបរទេស', 'និង', 'សហប្រតិបត្តិការ', 'អន្តរជាតិ', 'បាន']
        tags = ['B-ORG', 'O', 'O', 'O', 'B-ORG', 'O']
        self.assertEqual(expand_entity_boundaries(words, tags),
                         tags)  # ORG boundaries must come from the trained model.
        self.assertEqual(expand_entity_boundaries(
            ['នៅ', 'រាជធានី', 'ភ្នំពេញ'], ['O', 'O', 'B-LOC']),
            ['O', 'B-LOC', 'I-LOC'])
        self.assertEqual(expand_entity_boundaries(
            ['លោក', 'ហ៊ុន', 'ម៉ាណែត'], ['O', 'B-PER', 'I-PER']),
            ['B-PER', 'I-PER', 'I-PER'])
        self.assertEqual(expand_entity_boundaries(
            ['សុខា', 'និង', 'ដារ៉ា', 'ដែល'], ['B-PER', 'O', 'O', 'O']),
            ['B-PER', 'O', 'B-PER', 'O'])
        self.assertEqual(expand_entity_boundaries(
            ['អាជ្ញាធរ', 'ជាតិ', 'អប្សរា', 'ព្រមទាំង', 'អ្នកស្រាវជ្រាវ'],
            ['B-ORG', 'O', 'O', 'O', 'O']),
            ['B-ORG', 'O', 'O', 'O', 'O'])
        self.assertEqual(prepare_display_tags(
            ['។', 'សុខា', 'និង', 'ដារ៉ា', 'ដែល'],
            ['B-PER', 'B-PER', 'O', 'O', 'O']),
            ['O', 'B-PER', 'O', 'B-PER', 'O'])
        self.assertEqual(prepare_display_tags(
            ['លោក', 'ប៉ែន', 'សុវិជាតិ', '៖'],
            ['B-PER', 'B-PER', 'I-PER', 'I-PER']),
            ['B-PER', 'I-PER', 'I-PER', 'O'])
        # A nationality phrase is O; the same country as a place stays LOC.
        self.assertEqual(prepare_display_tags(
            ['ជនជាតិ', 'ឥណ្ឌូនេស៊ី', '១៨០', 'នាក់', 'ពី', 'កម្ពុជា', '។'],
            ['O', 'B-LOC', 'O', 'O', 'O', 'B-LOC', 'O']),
            ['O', 'O', 'O', 'O', 'O', 'B-LOC', 'O'])
        self.assertEqual(prepare_display_tags(
            ['ជនជាតិ', 'សហរដ្ឋ', 'អាមេរិក', 'និង', 'ស្ថានទូត', 'ឥណ្ឌូនេស៊ី'],
            ['B-LOC', 'I-LOC', 'I-LOC', 'O', 'B-ORG', 'I-ORG']),
            ['O', 'O', 'O', 'O', 'B-ORG', 'I-ORG'])
        self.assertEqual(prepare_display_tags(
            ['ជនជាតិ', 'ចិន', 'ទៅ', 'ប្រទេស', 'ចិន'],
            ['O', 'B-PER', 'O', 'O', 'B-LOC']),
            ['O', 'B-PER', 'O', 'B-LOC', 'I-LOC'])  # only LOC is cleared
        self.assertEqual(expand_entity_boundaries(
            ['ក្រសួង', 'ពាណិជ្ជកម្ម', 'សហការ', 'ជាមួយ'],
            ['B-ORG', 'I-ORG', 'O', 'O']),
            ['B-ORG', 'I-ORG', 'O', 'O'])

    def test_o_bias_decoder(self):
        from neam.inference import decode_probabilities
        labels = {i: label for i, label in enumerate(LABELS)}
        probabilities = [[0.55, 0.01, 0.01, 0.01, 0.01, 0.40, 0.01]]
        self.assertEqual(decode_probabilities(probabilities, labels, 1.0)[0], ['O'])
        self.assertEqual(decode_probabilities(probabilities, labels, 0.7)[0], ['B-LOC'])
        org_probabilities = [[0.35, 0.01, 0.01, 0.40, 0.01, 0.20, 0.02]]
        self.assertEqual(decode_probabilities(
            org_probabilities, labels, label_biases={'B-ORG': 0.7})[0], ['O'])

    def test_viterbi_decoding_keeps_valid_entity_paths(self):
        from neam.inference import decode_probabilities
        labels = {i: label for i, label in enumerate(LABELS)}
        #            O     B-PER I-PER B-ORG I-ORG B-LOC I-LOC
        rows = [[0.10, 0.00, 0.00, 0.80, 0.00, 0.10, 0.00],
                [0.10, 0.00, 0.00, 0.00, 0.45, 0.45, 0.00],
                [0.10, 0.00, 0.00, 0.00, 0.30, 0.00, 0.60]]
        # Word-by-word argmax needs repair: I-LOC cannot follow I-ORG.
        self.assertEqual(decode_probabilities(rows, labels)[0], ['B-ORG', 'I-ORG', 'B-LOC'])
        # Viterbi scores whole valid paths: 0.8*0.45*0.6 beats 0.8*0.45*0.3.
        self.assertEqual(decode_probabilities(rows, labels, decoding='viterbi')[0],
                         ['B-ORG', 'B-LOC', 'I-LOC'])
        rows[2] = [0.10, 0.00, 0.00, 0.00, 0.50, 0.00, 0.40]
        self.assertEqual(decode_probabilities(rows, labels, decoding='viterbi')[0],
                         ['B-ORG', 'I-ORG', 'I-ORG'])
        tags, scores = decode_probabilities([[0.3, 0, 0, 0, 0.7, 0, 0]], labels, decoding='viterbi')
        self.assertEqual(tags, ['O'])  # A sequence cannot start inside an entity.
        self.assertAlmostEqual(scores[0], 0.3)
        with self.assertRaises(ValueError):
            decode_probabilities(rows, labels, decoding='beam')

    def test_first_subword_only(self):
        self.assertEqual(align_labels([None, 0, 0, 1, 1, None], ['B-PER', 'O']),
                         [-100, 1, -100, 0, -100, -100])

    def test_first_content_subword_skips_bare_marker(self):
        from neam.preprocess import FIRST_CONTENT_SUBWORD, word_label_positions
        ids = [None, 0, 1, 1, 2, 2, 3, None]
        tokens = ['[CLS]', '▁ក្រសួង', '▁', 'សុខាភិបាល', '▁', '▁', '▁បាន', '[SEP]']
        self.assertEqual(word_label_positions(ids, tokens), {0: 1, 1: 2, 2: 4, 3: 6})
        self.assertEqual(word_label_positions(ids, tokens, FIRST_CONTENT_SUBWORD),
                         {0: 1, 1: 3, 2: 4, 3: 6})
        self.assertEqual(align_labels(ids, ['B-ORG', 'I-ORG', 'O', 'O'], tokens, FIRST_CONTENT_SUBWORD),
                         [-100, 3, -100, 4, 0, -100, 0, -100])
        with self.assertRaises(ValueError):
            word_label_positions(ids, None, FIRST_CONTENT_SUBWORD)

    def test_normalization_offsets_and_coeng(self):
        text = '\u200bក្\u200dខ ក្\u200cខ'
        words = segment(text, lambda t: t.split())
        self.assertEqual(normalize(text), 'ក្ខ ក្ខ')
        self.assertEqual([w.text for w in words], ['ក្ខ', 'ក្ខ'])
        for word in words:
            self.assertEqual(normalize(text[word.start:word.end]), word.text)

    def test_dropped_text_is_error(self):
        with self.assertRaises(ValueError):
            segment('abc', lambda t: ['a', 'c'])

    def test_bio_repair_and_entity_offsets(self):
        tags = repair_iob2(['I-PER', 'I-PER', 'I-LOC', 'O'])
        self.assertEqual(tags, ['B-PER', 'I-PER', 'B-LOC', 'O'])
        result = collect_entities('a b c', [Word('a', 0, 1), Word('b', 2, 3), Word('c', 4, 5)],
                                  ['B-PER', 'I-PER', 'B-LOC'], [.9, .7, .8])
        self.assertEqual(result[0]['text'], 'a b')
        self.assertAlmostEqual(result[0]['score'], .8)

    def test_entity_does_not_continue_onto_next_line(self):
        text = 'Ministry of Interior\nGeneral Department'
        words = [Word('Ministry', 0, 8), Word(' ', 8, 9), Word('of', 9, 11), Word(' ', 11, 12),
                 Word('Interior', 12, 20), Word('\n', 20, 21), Word('General', 21, 28),
                 Word(' ', 28, 29), Word('Department', 29, 39)]
        result = collect_entities(text, words, ['B-ORG'] + ['I-ORG'] * 8, [.9] * 9)
        self.assertEqual([e['text'] for e in result], ['Ministry of Interior', 'General Department'])

    def test_attached_khmer_punctuation_is_trimmed_from_entity(self):
        text = 'សុខា៖'
        result = collect_entities(text, [Word(text, 0, len(text))], ['B-PER'], [.9])
        self.assertEqual(result[0]['text'], 'សុខា')
        self.assertEqual(result[0]['end'], len(text) - 1)
        self.assertTrue(result[0]['boundary_adjusted'])

    def test_ceremonial_titles_and_whitespace_stay_outside_spans(self):
        def reconstruct(pieces, tags):
            text, words, start = ''.join(pieces), [], 0
            for piece in pieces:
                words.append(Word(piece, start, start + len(piece)))
                start += len(piece)
            shown = prepare_display_tags(pieces, tags)
            return [e['text'] for e in collect_entities(text, words, shown, [.9] * len(words), model_tags=tags)]
        self.assertEqual(reconstruct(['ឯកឧត្តម', ' ', 'កែវ', ' ', 'សុវណ្ណ'], ['B-PER'] + ['I-PER'] * 4),
                         ['កែវ សុវណ្ណ'])
        self.assertEqual(reconstruct(['ឯកឧត្ដម', 'សន្ដិ', 'បណ្ឌិត', 'ឆៃ'], ['B-PER'] + ['I-PER'] * 3), ['ឆៃ'])
        self.assertEqual(reconstruct(['សន្តិ', ' ', 'សុខា'], ['B-PER', 'I-PER', 'I-PER']), ['សន្តិ សុខា'])
        self.assertEqual(reconstruct(['លោក', ' ', 'សុខា'], ['B-PER', 'I-PER', 'I-PER']), ['លោក សុខា'])
        self.assertEqual(reconstruct(['Acme', ' Corp', ' ', 'បាន'], ['B-ORG', 'I-ORG', 'I-ORG', 'O']),
                         ['Acme Corp'])


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    raw = Tokenizer(WordPiece({'[UNK]': 0, '[PAD]': 1, '[CLS]': 2, '[SEP]': 3,
                              'a': 4, '##b': 5, 'c': 6}))
    raw.post_processor = TemplateProcessing(single='[CLS] $A [SEP]',
                                            special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    return PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', pad_token='[PAD]',
                                  cls_token='[CLS]', sep_token='[SEP]', model_max_length=32)


class IntegrationTests(unittest.TestCase):
    def test_whole_word_chunks_and_entity_boundary(self):
        tokenizer = tiny_tokenizer()
        words = ['ab'] * 7
        tags = ['O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O']
        chunks = encode_chunks(tokenizer, words, tags, max_length=8)
        self.assertEqual(chunks[0][1], 2)
        self.assertEqual(sum(end - start for start, end, _, _ in chunks), 7)
        self.assertEqual(sum(sum(x != -100 for x in chunk['labels']) for _, _, chunk, _ in chunks), 7)
        with self.assertRaises(ValueError):
            encode_chunks(tokenizer, words, ['B-PER'] + ['I-PER'] * 6, max_length=8)

    def test_train_save_reload_predict(self):
        import torch
        from transformers import BertConfig, BertForTokenClassification, DataCollatorForTokenClassification
        from neam.inference import KhmerNER
        from neam.preprocess import LABELS, LABEL2ID
        tokenizer = tiny_tokenizer()
        model = BertForTokenClassification(BertConfig(vocab_size=7, hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, num_labels=7,
            id2label=dict(enumerate(LABELS)), label2id=LABEL2ID))
        model.config.khmer_ner_max_length = 8
        chunk = encode_chunks(tokenizer, ['ab', 'c'], ['B-PER', 'O'], 8)[0][2]
        batch = DataCollatorForTokenClassification(tokenizer)([dict(chunk)])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            predictor = KhmerNER(directory, device='cpu')
            tags, scores = predictor.predict_words(['ab'] * 10)
            self.assertEqual(len(tags), 10)
            self.assertTrue(all(0 <= score <= 1 for score in scores))
            # A checkpoint without a recorded position keeps v12 first-subword inference.
            self.assertEqual(predictor.label_position, 'first-subword')
            model.config.khmer_ner_label_position = 'first-content-subword'
            model.save_pretrained(directory)
            predictor = KhmerNER(directory, device='cpu')
            self.assertEqual(predictor.label_position, 'first-content-subword')
            self.assertEqual(len(predictor.predict_words(['ab'] * 10)[0]), 10)
            single = predictor.predict_word_probabilities(['ab', 'c'])

            # An ensemble of two copies averages to the same probabilities and
            # takes decoding and bias overrides from ensemble.json.
            with tempfile.TemporaryDirectory() as ensemble_dir:
                Path(ensemble_dir, 'ensemble.json').write_text(json.dumps(
                    {'members': [directory, directory], 'decoding': 'viterbi', 'o_bias': 0.8}))
                ensemble = KhmerNER(ensemble_dir, device='cpu')
                self.assertEqual(len(ensemble.models), 2)
                self.assertEqual((ensemble.decoding, ensemble.o_bias), ('viterbi', 0.8))
                for a, b in zip(single, ensemble.predict_word_probabilities(['ab', 'c'])):
                    for x, y in zip(a, b):
                        self.assertAlmostEqual(x, y, places=6)
                Path(ensemble_dir, 'ensemble.json').write_text(json.dumps({'members': [directory], 'oops': 1}))
                with self.assertRaises(ValueError):
                    KhmerNER(ensemble_dir, device='cpu')

            # Overlapping windows cover every word; a document that fits in one
            # window gives the same probabilities as disjoint windows.
            predictor.context_windows = 'overlap'
            self.assertEqual(len(predictor.predict_word_probabilities(['ab', 'c'] * 9)), 18)
            for a, b in zip(single, predictor.predict_word_probabilities(['ab', 'c'])):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=6)

    def test_overlapping_context_windows(self):
        from neam.preprocess import _word_piece_counts, encode_context_windows
        tokenizer = tiny_tokenizer()
        words = ['ab', 'c', 'c', 'ab', 'c', 'ab', 'ab', 'c', 'c', 'c', 'ab', 'c', 'ab']
        counts = _word_piece_counts(tokenizer, words)
        windows = encode_context_windows(tokenizer, words, max_length=8)
        owners = [word for *_, owned in windows for word in owned]
        self.assertEqual(sorted(owners), list(range(len(words))))  # each word read exactly once
        self.assertTrue(all(len(chunk['input_ids']) <= 8 for _, _, chunk, _, _ in windows))
        spans = [(start, end) for start, end, *_ in windows]
        self.assertTrue(all(b[0] < a[1] for a, b in zip(spans, spans[1:])))  # windows overlap
        self.assertEqual(spans[-1][1], len(words))
        # The right-aligned final window cannot take another word without exceeding the budget.
        self.assertGreater(sum(counts[spans[-1][0] - 1:]), 8 - 2)

        def margin(word, start, end):
            left = sum(counts[start:word]) if start > 0 else float('inf')
            right = sum(counts[word + 1:end]) if end < len(words) else float('inf')
            return min(left, right)
        for start, end, _, _, owned in windows:
            for word in owned:
                best = max(margin(word, s, e) for s, e in spans if s <= word < e)
                self.assertEqual(margin(word, start, end), best)
        single = encode_context_windows(tokenizer, ['ab', 'c'], max_length=8)
        self.assertEqual([(s, e, list(o)) for s, e, _, _, o in single], [(0, 2, [0, 1])])

    def test_real_khmer_segmenter(self):
        text = 'សុខារស់នៅភ្នំពេញ។'
        words = segment(text)
        self.assertTrue(words)
        self.assertEqual(''.join(w.text for w in words), text)


class AppTests(unittest.TestCase):
    def test_app_builds_with_url_import_only(self):
        import app
        demo = app.create_app('unused-model-path')  # the model loads lazily on first prediction
        labels = {getattr(block, 'label', None) for block in demo.blocks.values()}
        self.assertIn('Public article URL', labels)
        self.assertFalse(any('PDF' in str(label) for label in labels))


if __name__ == '__main__':
    unittest.main()
