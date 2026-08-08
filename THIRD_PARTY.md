# Third-party components

DeepTrace-R1 separates direct dependencies, selectively ported
code, and architectural references. A repository appearing here does not mean
its code has already been copied or its functionality has been implemented.

## Pinned Phase 0 components

| Component | Repository | Pinned commit | License | Intended reuse | Current reuse |
| --- | --- | --- | --- | --- | --- |
| Open Deep Research | https://github.com/langchain-ai/open_deep_research | `d337ae32ed4ff8f4c6fbe192ba3bf1b2d6610799` | MIT | Selective workflow reference/port | No source files copied |
| Agent-R1 | https://github.com/AgentR1/Agent-R1 | `b124aa46534cbf2fb8bc8af11405774984c42ac7` | MIT | Dependency and training recipes | Not installed; no source files copied |
| veRL | https://github.com/verl-project/verl | `f9c855f7cf04d603c9546bc01776c74806a879c1` (`v0.7.0`) | Apache-2.0 | External training dependency required by pinned Agent-R1 | Not installed; no source files copied |

The Open Deep Research and Agent-R1 commits above were returned by
`git ls-remote <repository> HEAD` on the target server on 2026-07-29.
Agent-R1's pinned README explicitly requires `verl==0.7.0`, so veRL is pinned
to the target-server-resolved `v0.7.0` tag rather than an incompatible moving
HEAD. License files were read at the exact commits. These records do not claim
runtime validation.

## Pinned reference-only repositories

| Component | Repository | Pinned commit | License | Planned use |
| --- | --- | --- | --- | --- |
| Search-R1 | https://github.com/PeterGriffinJin/Search-R1 | `598e61bd1d36895726d28a8d06b3a15bed19f5d3` | Apache-2.0 | Retriever server and search-agent baseline reference |
| GPT Researcher | https://github.com/assafelovic/gpt-researcher | `5d84d2f5553e70a2765a8ff3a0d2672d60437ce8` | Apache-2.0 | Provider adapters, export and observability reference |
| Tongyi DeepResearch | https://github.com/Alibaba-NLP/DeepResearch | `f72f75d8c3eb842f2bbbab096a12206ff66e270f` | Apache-2.0 | Tool protocol and evaluation reference |

These repositories remain reference-only. Pinning a commit does not authorize
copying code. Any later selective port must record the reused files, notices,
modifications, and project-specific tests.

## Reuse rules

- Keep upstream copyright and license headers.
- Mark modifications to Apache-2.0 source files.
- Record every copied or modified upstream file in `upstream.lock.yaml`.
- Prefer package dependencies for trainers and infrastructure.
- Do not describe upstream work as DeepTrace-authored work.
- Do not use code, weights, or datasets with unclear licenses.
- Never commit API keys, cookies, access tokens, model weights, or private
  server coordinates.

## Evaluation datasets

The project stores a six-record smoke subset and a 90-record final tool-loop
subset. These are data sources rather than vendored software:

| Dataset | Upstream | Data license |
|---|---|---|
| HotpotQA | https://github.com/hotpotqa/hotpot | CC BY-SA 4.0 |
| 2WikiMultiHopQA | https://github.com/Alab-NII/2wikimultihop | Apache 2.0 |
| MuSiQue | https://github.com/stonybrooknlp/musique | CC BY 4.0 |

Exact mirror revisions and attribution details are in
`data/evaluation/agentic-search-v1/SOURCES.md` and
`data/evaluation/final-tool-loop-test-v1/manifest.json`. The HotpotQA-derived
records remain subject to CC BY-SA 4.0 and are not relicensed by this project.
