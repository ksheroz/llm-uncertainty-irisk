# Uncertainty Quantification in Large Language Models

Research code for the **UQ in LLMs** project at the [iRisk Lab](https://asrm.illinois.edu/), University of Illinois Urbana-Champaign, Fall 2026.

### Author

Sheroz Khan, supervised by Prof. Zhiyu (Frank) Quan.

### Abstract

Large language models can generate confident-sounding outputs that are factually wrong. This project investigates uncertainty quantification methods that detect such failures, starting with semantic entropy (Farquhar et al., Nature 2024) and extending to domain-shift-aware conformal prediction for specialized (actuarial, insurance) text.

### Repository Structure

```
irisk-uq-llm/
├── configs/              # Experiment configs (model, hyperparameters)
├── data/
│   ├── raw/              # Raw question sets, API responses
│   └── processed/        # Cleaned datasets with labels
├── doc/                  # Paper notes, meeting notes
├── notebooks/
│   ├── weekly/           # Weekly notebooks submitted to Prof. Quan
│   └── exploration/      # Scratch analysis and prototyping
├── reports/
│   ├── figures/          # Generated plots and diagrams
│   └── presentations/    # Slide decks
├── scripts/              # Standalone run scripts
├── src/
│   ├── generation/       # LLM sampling (Ollama, API wrappers)
│   ├── clustering/       # NLI-based semantic clustering
│   ├── evaluation/       # Metrics, comparison frameworks
│   └── utils/            # Shared helpers
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Setup

```bash
git clone https://github.com/ksheroz/irisk-uq-llm.git
cd irisk-uq-llm
pip install -r requirements.txt

# Local LLM via Ollama (https://ollama.com/download)
ollama pull gemma3:4b
```

### Weekly Progress

| Week | Notebook | Focus |
|------|----------|-------|
| 1 | `notebooks/weekly/week01_semantic_entropy.ipynb` | Farquhar et al. (2024): math, implementation, pilot |
| 2 | `notebooks/weekly/week02_se_critical_evaluation.ipynb` | Real generations, hard questions, SE vs naive baseline |
| 3 | | Token-level log-prob SE variant, larger benchmark |

### Key References

- Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y. (2024). Detecting Hallucinations in Large Language Models Using Semantic Entropy. *Nature*, 630, 625-630.
- Kuhn, L., Gal, Y., and Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. *ICLR 2023*.

### License

MIT
