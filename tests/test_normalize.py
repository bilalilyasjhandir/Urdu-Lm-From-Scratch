"""Run with:  python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from urdu_lm.filters import FilterConfig, arabic_share, function_word_share, reject_reason  # noqa: E402
from urdu_lm.normalize import ZWNJ, normalize  # noqa: E402

YEH_AR, YEH_UR = "ي", "ی"
KAF_AR, KAF_UR = "ك", "ک"
HEH_AR, HEH_UR = "ه", "ہ"
HAI = "ہے"          # hai
KE = "کے"           # ke
MEIN = "میں"   # mein
KITAB = "کتاب"  # kitab


class NormalizeTests(unittest.TestCase):
    def test_arabic_lookalikes_become_urdu(self):
        self.assertEqual(normalize(YEH_AR + KAF_AR + HEH_AR), YEH_UR + KAF_UR + HEH_UR)
        self.assertEqual(normalize("ى"), YEH_UR)  # alef maksura

    def test_digits_become_ascii(self):
        self.assertEqual(normalize("١٢٣ ۴۵۶"), "123 456")

    def test_vowel_marks_removed(self):
        # kaf + zabar + teh + shadda -> kaf + teh
        self.assertEqual(normalize("کَتّ"), "کت")

    def test_tatweel_and_invisibles_removed(self):
        self.assertEqual(normalize("کــت​‎﻿اب"), KITAB)

    def test_presentation_forms_folded(self):
        # U+FEDB is "kaf initial form"; NFKC gives Arabic kaf, the table gives keheh
        self.assertEqual(normalize("ﻛ"), KAF_UR)

    def test_hamza_composes_after_mapping(self):
        # Arabic heh + hamza above -> heh goal + hamza above -> single U+06C2
        self.assertEqual(normalize(HEH_AR + "ٔ"), "ۂ")
        # alef + madda -> alef with madda above
        self.assertEqual(normalize("آ"), "آ")

    def test_zwnj_kept_only_between_letters(self):
        inside = "خوب" + ZWNJ + "صورت"
        self.assertEqual(normalize(inside), inside)
        self.assertEqual(normalize(inside.replace(ZWNJ, ZWNJ * 3)), inside)
        self.assertEqual(normalize(ZWNJ + KITAB + ZWNJ + " " + ZWNJ + HAI), KITAB + " " + HAI)

    def test_urls_and_emails_removed(self):
        out = normalize(f"{KITAB} https://example.com/a?b=1 {HAI} someone@example.com {KE}")
        self.assertEqual(out, f"{KITAB} {HAI} {KE}")

    def test_whitespace(self):
        self.assertEqual(normalize(f"  {KITAB}\t\t{HAI} \r\n\r\n\n\n {KE} {MEIN}  "),
                         f"{KITAB} {HAI}\n\n{KE} {MEIN}")

    def test_idempotent(self):
        messy = (f"﻿{KITAB}َ {YEH_AR}{HEH_AR}ٔ ١٢ www.x.com \r\n\n\n"
                 f"{ZWNJ}{HAI}{ZWNJ}{ZWNJ}{KE}ـ ﷺ")
        once = normalize(messy)
        self.assertEqual(normalize(once), once)


class FilterTests(unittest.TestCase):
    def test_shares(self):
        text = f"{KITAB} {MEIN} {HAI} abc"
        self.assertAlmostEqual(arabic_share(text), 9 / 12)
        self.assertAlmostEqual(function_word_share(text), 2 / 4)
        self.assertAlmostEqual(function_word_share(f"{HAI}۔ {KITAB}"), 0.5)  # trailing full stop

    def test_reject_reasons(self):
        cfg = FilterConfig(min_chars=20)
        urdu = " ".join([KITAB, MEIN, HAI, KE] * 5)
        self.assertIsNone(reject_reason(urdu, cfg))
        self.assertEqual(reject_reason(KITAB, cfg), "too_short")
        self.assertEqual(reject_reason("hello world " * 5 + KITAB, cfg), "low_arabic_share")
        self.assertEqual(reject_reason(" ".join([KITAB] * 20), cfg), "low_function_word_share")


if __name__ == "__main__":
    unittest.main()
