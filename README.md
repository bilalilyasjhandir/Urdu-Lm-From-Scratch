# Urdu language model from scratch: a tokenizer study

**Status: in progress.** The corpus is built. The tokenizers, the model and the training loop are written and tested at toy scale on CPU. No full model has been trained yet, so there are no results here.

## The question

With the model, the training text and the compute held fixed, how much does the tokenizer matter for an Urdu language model, and which kind works best?

## Why this question

Two recent papers train Urdu models from scratch, and both vary only the vocabulary size of one tokenizer algorithm:

- **UrduLM** (Naqvi et al., 2026) trains a decoder-only model of about 100M parameters with a custom BPE tokenizer, at 10k, 20k and 32k vocabularies.
- **DunbaaBERT** (Maab et al., 2026) trains RoBERTa-base encoders with byte-level BPE at 32k, 52k and 96k.

Work on other morphologically rich languages reports that the algorithm matters too. Hou et al. (2023) found that morphological segmentation gave lower perplexity than BPE in GPT and BERT models. Vemula et al. (2025) found that Unigram beat BPE in most settings on Telugu. I did not find that comparison for Urdu. This project runs it.

It also uses only data anyone can download. UrduLM's corpus is public only as a preview, so its numbers cannot be reproduced from scratch.

## Design

One decoder-only transformer of about 100M parameters is trained once per tokenizer. Everything else stays the same.

| Arm | Tokenizer | Vocabulary |
|---|---|---|
| `bpe-32k` | SentencePiece BPE (the setup of earlier Urdu work) | 32,000 |
| `unigram-32k` | SentencePiece Unigram | 32,000 |
| `morph-bpe-32k` | Unsupervised morphological segmentation (Morfessor), then BPE inside each morph | 32,000 |
| `char` | One token per character | about 190, set from the corpus |
| `bpe-8k`, `bpe-16k`, `bpe-64k` | Vocabulary-size sweep | 8,000 to 64,000 |

**Model.** 12 layers, 12 heads, width 768, context of 1,024 tokens ([urdu_lm/model.py](urdu_lm/model.py)). The body has 85M parameters in every arm. The embedding table grows with the vocabulary, so the total is about 110M at 32,000 tokens.

**What is held fixed.** Every arm gets the same compute: 38,000 optimiser steps of 64 sequences of 1,024 tokens, which is 2.49 billion token positions, with the same optimiser settings ([configs/base.json](configs/base.json)). Training windows are drawn at random from the token stream. The budget is smaller than every subword arm's training set (2.82 to 3.23 billion tokens), so no arm makes a full pass. An arm with shorter tokens reads less text in the same budget, which is the real cost of choosing it: the character arm reads about a fifth of what the others do. The text read by each arm is reported next to its score. One learning rate is used for all arms and was not tuned per arm.

**Same text for every arm.** A fixed alphabet (the characters covering 99.99% of the training text) is chosen once. Every other character is replaced by one placeholder before any tokenizer sees the text, so all arms model exactly the same character sequence.

**Morphology arm.** Morfessor is unsupervised, and two settings were chosen by reading its output on training words. It is trained on word types, because with frequency-weighted counts it left every frequent word whole. A split is used only if its first morph has at least three letters, because otherwise it also cut short function words into single letters. Neither setting was tuned on validation or test data. `reports/morph_examples.md` shows what it produces.

**Seeds.** The main arms are repeated over several seeds, so differences between arms can be compared with run-to-run noise.

### Evaluation

- **Bits per character** on held-out websites. Loss per token cannot be compared between tokenizers, because a model with longer tokens predicts more text per step. Bits per character can.
- **MultiBLiMP, Urdu part** (Jumelet et al., 2025): 550 sentence pairs that differ in one agreement feature. The model should give the grammatical sentence the higher probability. This set is small. In 143 pairs the source treebank joins multiword tokens with an underscore, which never occurs in running Urdu, so the underscore is replaced by a space in both sentences.
- **UrBLiMP** (Adeeba et al., 2026): 5,696 pairs over ten phenomena. The paper says the data will be released. I have not found it yet, so this is not confirmed.
- **Reference point:** the released UrduLM checkpoint, scored on the same test text.

## Data

Source: the Urdu subset of [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (`urd_Arab`, ODC-By), 4.86 million web documents, already language-identified and near-deduplicated by its authors.

**Normalisation** ([urdu_lm/normalize.py](urdu_lm/normalize.py)). The same Urdu word can be typed with different Unicode code points, which would split one word into several vocabulary entries and affect each tokenizer differently.

- Arabic look-alike letters are mapped to their Urdu code points (yeh, kaf, heh, teh marbuta).
- Presentation forms are folded to plain letters (NFKC).
- Arabic-Indic digits become ASCII digits.
- Optional vowel marks, tatweel and invisible formatting characters are removed.
- The zero-width non-joiner is kept only between two letters.
- URLs and email addresses are removed, and whitespace is collapsed.

**Filters** ([urdu_lm/filters.py](urdu_lm/filters.py)). A document is dropped if it has fewer than 200 characters, if under 60% of its characters are Arabic script, or if under 5% of its words are common Urdu function words. Thresholds were set by reading borderline documents from a 12,000-document sample.

**Split by website.** About 0.5% of sites go to validation and 0.5% to test, and no page from those sites is trained on. Pages from one site share templates and writers, so a random document split would leak near-copies of test pages into training.

**Result of the full run** ([reports/cleaning_report.json](reports/cleaning_report.json), [reports/corpus_stats.md](reports/corpus_stats.md)). Of 4,858,678 documents, 4,849,117 (99.8%) passed the filters and 141 exact duplicates were removed.

| Split | Documents | Characters | Words | Websites |
|---|---:|---:|---:|---:|
| Train | 4,818,016 | 12.06 billion | 2.57 billion | 29,069 |
| Validation | 2,600 | 7.2 million | 1.5 million | 157 |
| Test | 2,152 | 7.0 million | 1.5 million | 173 |

Validation and test are capped at 50 documents per website, so no single site dominates them. The largest training site supplies 4.6% of the characters and the largest ten supply 18.7%.

## Tokenizers on the full corpus

From [reports/tokenizers.json](reports/tokenizers.json) and [reports/encoding.json](reports/encoding.json). All seven reproduce the validation text exactly after decoding, and none produces an unknown token. The alphabet has 186 characters.

| Arm | Vocabulary | Tokens per word | Characters per token | Training tokens |
|---|---:|---:|---:|---:|
| `bpe-8k` | 8,000 | 1.290 | 3.64 | 3.23 billion |
| `bpe-16k` | 16,000 | 1.199 | 3.92 | 3.01 billion |
| `bpe-32k` | 32,000 | 1.147 | 4.09 | 2.89 billion |
| `bpe-64k` | 64,000 | 1.119 | 4.20 | 2.82 billion |
| `unigram-32k` | 32,000 | 1.144 | 4.11 | 2.88 billion |
| `morph-bpe-32k` | 32,000 | 1.201 | 3.91 | 3.02 billion |
| `char` | 190 | 4.697 | 1.00 | 12.07 billion |

Tokens per word and characters per token are measured on the validation set. In the morphology arm, Morfessor's splits are used for 7.4% of running words, giving 1.08 morphs per word ([reports/morph_examples.md](reports/morph_examples.md)).

## Training speed

Measured with real training steps on each card ([reports/calibration.json](reports/calibration.json)). The two cards perform the same.

| Arm | Micro-batch | Tokens per second | Peak memory | One full run |
|---|---:|---:|---:|---:|
| `bpe-32k` | 16 | 59,800 | 12.5 GB | about 11.6 hours |
| `bpe-64k` | 16 | 47,700 | 18.6 GB | about 14.5 hours |
| `char` | 16 | 76,200 | 6.4 GB | about 9.1 hours |

Planned: the four main arms (`bpe-32k`, `unigram-32k`, `morph-bpe-32k`, `char`) with three seeds each, and one run each for `bpe-8k`, `bpe-16k` and `bpe-64k`. That is 15 runs, roughly 170 GPU-hours.

## Running it

Python 3.10 or newer. Install PyTorch with CUDA from pytorch.org first, then:

```
pip install -r requirements.txt
python scripts/00_check_env.py
python scripts/01_download.py --data-dir D:/urdu-lm-data
python scripts/02_clean.py    --data-dir D:/urdu-lm-data
python scripts/03_stats.py    --data-dir D:/urdu-lm-data
python scripts/04_tokenizer_data.py   --data-dir D:/urdu-lm-data
python scripts/05_train_tokenizers.py --data-dir D:/urdu-lm-data
python scripts/06_encode.py           --data-dir D:/urdu-lm-data
python scripts/07_calibrate.py        --data-dir D:/urdu-lm-data --gpu 0
```

One training run is one tokenizer arm, one seed, one GPU:

```
python scripts/train.py --data-dir D:/urdu-lm-data --arm bpe-32k --seed 1 --gpu 0
```

`scripts/run_queue.py` chains several runs on one GPU, and `scripts/evaluate.py` scores every finished run and writes `reports/results.md`. Every script can be re-run after an interruption and continues where it stopped. A stopped training run resumes from its last checkpoint and trains on exactly the data it would have seen. Add `--sample` to the download for a 71 MB dry run. Tests: `python -m unittest discover -s tests -v`.

## Hardware

Two NVIDIA RTX 3090 cards (24 GB each), Windows 10, PyTorch 2.9.1. Each card trains one arm at a time, so two arms run side by side.

## Progress

- [x] Literature and data check
- [x] Download, normalisation, filtering, website-level split, corpus statistics
- [x] Full data run on the training machine
- [x] Tokenizers, token encoding, model and training loop (tested at toy scale)
- [x] Tokenizers and encoding on the full corpus, speed measurement
- [x] Evaluation script (tested at toy scale)
- [ ] Training runs
- [ ] Results and write-up

## References

- Naqvi, Sajid, Haider, Asad, Fatima and Samad (2026). *UrduLM: A Resource-Efficient Monolingual Urdu Language Model.* [arXiv:2601.17664](https://arxiv.org/abs/2601.17664)
- Maab, Jamil and Schmitt (2026). *DunbaaBERT: From Sacrifice to Semantics.* [arXiv:2605.26935](https://arxiv.org/abs/2605.26935)
- Hou, Katinskaia, Vu and Yangarber (2023). *Effects of sub-word segmentation on performance of transformer language models.* [arXiv:2305.05480](https://arxiv.org/abs/2305.05480)
- Vemula, Dandapat, Sharma and Krishnamurthy (2025). *Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment.* [arXiv:2508.08424](https://arxiv.org/abs/2508.08424)
- Adeeba, Dillon, Sajjad and Bhatt (2026). *UrBLiMP: A Benchmark for Evaluating the Linguistic Competence of Large Language Models in Urdu.* [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.29/)
- Jumelet, Weissweiler, Nivre and Bisazza (2025). *MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs.* [arXiv:2504.02768](https://arxiv.org/abs/2504.02768)
- Penedo et al. (2025). *FineWeb2: One Pipeline to Scale Them All.* [arXiv:2506.20920](https://arxiv.org/abs/2506.20920)
