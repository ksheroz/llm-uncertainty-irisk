# Meeting Notes

## Week 1
- Prof. Quan assigned UQ in LLMs direction
- Focus: depth over breadth, understand the math, verify claims vs. implementation
- Deliverable: Jupyter notebook, consolidated
- Key paper: Farquhar et al. (Nature, 2024) on semantic entropy

## Week 2
- Could not present Week 1 due to time constraints; presenting Weeks 1+2 together
- Ollama/Gemma3:4b temperature issue: model ignores temperature param. Fix: custom Modelfile.
- Extended experiment: 5 easy + 10 hard questions
- Added baseline comparison: naive string entropy vs. semantic entropy
- Key finding: NLI clustering removes paraphrase noise on easy questions (high naive H -> zero SE)
- Key finding: SE flags genuine disagreement on hard/ambiguous questions
- Key limitation: SE = 0 when model is confidently wrong (measures consistency, not correctness)
