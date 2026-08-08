# Agentic Search v1 evaluation sources

This directory contains a deliberately small frozen evaluation subset. It is
used only to validate the Phase 4 evaluation pipeline and is not a statistically
representative benchmark.

## HotpotQA

- Upstream: https://github.com/hotpotqa/hotpot
- Homepage: https://hotpotqa.github.io/
- Mirror: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- Mirror revision:
  `1908d6afbbead072334abe2965f91bd2709910ab`
- License: CC BY-SA 4.0
- Selection: first two `distractor/validation` records

The two derived records and their supporting-fact text remain available under
CC BY-SA 4.0. Attribution:

> Yang, Zhilin; Qi, Peng; Zhang, Saizheng; Bengio, Yoshua; Cohen, William W.;
> Salakhutdinov, Ruslan; Manning, Christopher D. HotpotQA: A Dataset for
> Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.

## 2WikiMultiHopQA

- Upstream: https://github.com/Alab-NII/2wikimultihop
- Mirror: https://huggingface.co/datasets/framolfese/2WikiMultihopQA
- Mirror revision:
  `fe713bfbd1afbca1a65246741a75890405d56a3a`
- License: Apache 2.0
- Selection: first two `default/validation` records

Attribution:

> Ho, Xanh; Nguyen, Anh-Khoa Duong; Sugawara, Saku; Aizawa, Akiko.
> Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of
> Reasoning Steps. COLING 2020.

## MuSiQue

- Upstream: https://github.com/stonybrooknlp/musique
- Mirror: https://huggingface.co/datasets/bdsaglam/musique
- Mirror revision:
  `22873a405dd809893b22ada0b499299fb612d2df`
- License: CC BY 4.0
- Selection: first two `answerable/validation` records

Attribution:

> Trivedi, Harsh; Balasubramanian, Niranjan; Khot, Tushar; Sabharwal,
> Ashish. MuSiQue: Multihop Questions via Single-hop Question Composition.
> TACL 2022.

## Selection and tuning boundary

`test.jsonl` is frozen by the SHA-256 value in `manifest.json`. The first two
rows at each pinned revision were selected before running the Phase 4
comparison. Verifier thresholds and scripted mode policies must not be changed
using these six records.

`chinese-draft.jsonl` is authored for SearchLab and marked `reviewed=false`.
It is a schema and human-review draft only and is excluded from formal metrics.
