"""Run with:  python -m unittest discover -s tests -v"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urdu_lm.tokenizer import EOD_ID, JOINER, NL_ID, REPLACEMENT, SPECIALS, WORD_RE, Tokenizer  # noqa: E402
from urdu_lm.training import lr_at, sample_step  # noqa: E402

KITAB = "کتاب"  # kitab
HAI = "ہے"                # hai


def char_tokenizer(folder: Path, alphabet: str) -> Tokenizer:
    chars = sorted(alphabet) + [REPLACEMENT]
    meta = {"name": "char", "kind": "char", "alphabet": alphabet, "chars": chars,
            "vocab_size": len(SPECIALS) + len(chars)}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return Tokenizer(folder)


class CharTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tok = char_tokenizer(Path(self.tmp.name), "".join(sorted(set(KITAB + HAI + " 1"))))

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_with_line_breaks(self):
        text = f"{KITAB} {HAI}\n\n{HAI} 1"
        ids = self.tok.encode(text)
        self.assertEqual(ids[0], EOD_ID)
        self.assertEqual(int((ids == NL_ID).sum()), 2)
        self.assertEqual(self.tok.decode(ids), text)

    def test_unknown_characters_become_replacement(self):
        ids, chars = self.tok.encode_docs([f"{KITAB} x"])
        self.assertEqual(self.tok.decode(ids), f"{KITAB} {REPLACEMENT}")
        self.assertEqual(chars, len(KITAB) + 2 + 1)   # text plus one for <eod>

    def test_character_count_is_tokens_for_char_arm(self):
        docs = [f"{KITAB} {HAI}", f"{HAI}\n{KITAB}"]
        ids, chars = self.tok.encode_docs(docs)
        self.assertEqual(len(ids), chars)
        self.assertEqual(self.tok.decode(ids), "\n\n".join(docs))


class WordPatternTests(unittest.TestCase):
    def test_words_are_letter_runs_only(self):
        self.assertEqual(WORD_RE.findall(f"{KITAB}۔ 12 {HAI}‌{KITAB}, abc"), [KITAB, HAI, KITAB])

    def test_joiner_is_not_a_letter(self):
        self.assertEqual(WORD_RE.findall(KITAB + JOINER + HAI), [KITAB, HAI])


class TrainingHelperTests(unittest.TestCase):
    def test_batches_depend_only_on_seed_and_step(self):
        data = np.arange(5000, dtype=np.uint16)
        x1, y1 = sample_step(data, seed=3, step=17, sequences=8, block_size=64)
        x2, y2 = sample_step(data, seed=3, step=17, sequences=8, block_size=64)
        x3, _ = sample_step(data, seed=3, step=18, sequences=8, block_size=64)
        self.assertTrue((x1 == x2).all() and (y1 == y2).all())
        self.assertFalse((x1 == x3).all())
        self.assertTrue((y1 == x1 + 1).all())   # targets are the inputs shifted by one

    def test_learning_rate_schedule(self):
        cfg = {"lr": 1.0, "min_lr_ratio": 0.1, "warmup_steps": 10, "max_steps": 110}
        self.assertAlmostEqual(lr_at(0, cfg), 0.1)
        self.assertAlmostEqual(lr_at(9, cfg), 1.0)
        self.assertAlmostEqual(lr_at(60, cfg), 0.55)
        self.assertAlmostEqual(lr_at(110, cfg), 0.1)


if __name__ == "__main__":
    unittest.main()
