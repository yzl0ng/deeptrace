# Phase 5 trajectory pilot v1

- Status: `succeeded`
- Teacher calls: 100
- Accepted after deterministic refilter: 99
- Rejected: 1 (`failure-005`, seed query mismatch)
- Acceptance rate: 99%
- Total Teacher tokens: 149,159
- Duplicate IDs or record hashes: 0

The original filter incorrectly required a non-empty observation on the
terminal `answer` action. The raw responses were preserved, the schema was
corrected so only non-terminal actions require observations, and all 100
responses were deterministically refiltered without new API calls.

The pinned Qwen3-8B tokenizer measured 171–861 tokens per accepted record
(median 421); no record exceeded 2,048 tokens. The accepted records were split
deterministically into 89 training and 10 validation examples.

Raw Teacher content remains on the private training server. This repository
stores counts, hashes, and bounded metadata only. A 99% schema acceptance rate
does not establish semantic correctness or training benefit.
