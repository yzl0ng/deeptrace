# Phase 5 Base versus 10-step SFT tool-format evaluation

- Status: `succeeded_no_improvement`
- Frozen test cases: 6
- Base strict JSON trajectory validity: 0/6
- SFT strict JSON trajectory validity: 0/6
- Base mean latency: 5.02 seconds
- SFT mean latency: 9.16 seconds

Both modes used the same system prompt, question, greedy decoding, and
640-token limit. The Base and step-10 adapter both produced ordinary natural
language answers rather than the trained JSON trajectory schema. Inspection of
sample predictions confirmed this is not a parser or markdown-fence issue.

The training pipeline succeeded technically, but ten optimizer steps did not
teach the desired output contract. No quality improvement is claimed.

These six frozen test cases have now been observed. They must not be reused to
tune training duration or prompts and then presented as an untouched final
test. Before another quality-training run, complete the private human review
packet and use a separate dev set for choices; reserve a new independent test
for the final comparison.
