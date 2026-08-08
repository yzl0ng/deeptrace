# Phase 5 trajectory expansion 500 v1

- Status: `succeeded`
- Public training-split seeds: 500
- Unique IDs / queries: 500 / 500
- DeepSeek Teacher calls: 500
- Teacher tokens: 1,028,098
- Deterministic refilter: 479 accepted / 21 rejected
- Acceptance rate: 95.8%
- Duplicate accepted IDs or trajectory hashes: 0
- Frozen six-case test question overlap: 0
- SFT split: 431 train / 48 validation

The seeds came from the pinned training splits of HotpotQA,
2WikiMultiHopQA, and MuSiQue. The direct Hugging Face endpoint was unreachable,
so the same immutable revisions were read through `hf-mirror.com`; response
metadata matched the pinned commits.

Generation wrote and flushed each record immediately and can resume from
processed IDs after a network interruption. The initial online filter rejected
short answers such as `No`; this was a schema bug because short benchmark
answers are valid. Saved raw outputs were refiltered without new API calls.
The final 21 rejects are due to too many steps, malformed field types, missing
observations, invented evidence IDs, or seed-ID mismatch.

All 479 accepted records load through the real Qwen3 tokenizer and veRL
MultiTurnSFTDataset. Maximum sequence length is 1,005 and every record has
non-zero assistant loss tokens. A deterministic private review packet contains
five accepted records per source plus ten rejects. Its review fields remain
null; no human semantic approval is claimed.
