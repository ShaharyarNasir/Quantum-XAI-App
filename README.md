# Quantum XAI App: Non-Invasive Entanglement Analysis via Neural Proxies and SHAP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138279.svg)](https://doi.org/10.5281/zenodo.17138279)

This repository implements the framework from the paper "Probing Quantum Dynamics Through Explainable AI: Non-Invasive Entanglement Analysis via Neural Proxies and SHAP". It enables simulation of entangled qubit dynamics using QuTiP, training classical neural proxies (PyTorch), SHAP interpretability, and an interactive Gradio app for real-time exploration.

## Key Features
- **Simulations**: 2-6 qubit linear chain Hamiltonians with optional Lindblad decoherence (γ=0.05).
- **Neural Proxies**: ReLU networks (e.g., 20→10→1 for 2-qubit) trained with Adam (lr=0.01, MSE loss).
- **SHAP Analysis**: DeepExplainer reveals temporal impacts (e.g., peaks at t≈2.5/7.5 for 2-qubit).
- **Gradio Mini-App**: Interactive dashboard for parameter tuning (qubits, time, epochs) and visualizations (loss, correlations, SHAP beeswarm).
- **Scalability Script**: `scripts/scalability_and_shap_6qubit.py` for 6-qubit extension (MSE=0.0361, sub-quadratic fit).

## Setup
1. Clone repo: `git clone https://github.com/ShaharyarNasir/Quantum-XAI-App.git`
2. Install dependencies: `pip install qutip torch shap gradio matplotlib numpy scipy`
3. Run app: `python app.py` (opens at http://127.0.0.1:7860)

## Usage
- **Interactive Mode**: Launch Gradio app, adjust sliders (e.g., N=4, γ=0.05), click "Run Simulation" for 2x2 dashboard.
- **Offline Scalability**: Run `python scripts/scalability_and_shap_6qubit.py` for Fig. 5 outputs (`scalability_mse.pdf`, `shap_6qubit_inset.pdf`).
- **Reproduce Results**: See `notebooks/2qubit_validation.ipynb` for full 2/4-qubit runs (MSE=0.0196/0.025).

## Results Overview
| Qubits (N) | MSE | R² | Key SHAP Times (t) | Notes |
|------------|-----|----|---------------------|-------|
| 2          | 0.0196 | 0.984 | 2.5, 7.5 | Baseline |
| 4          | 0.025 | 0.975 | 2.0, 6.5, 8.0 | Linear chain |
| 6          | 0.0361 | 0.087 | 1.21, 2.63, 4.04 | Sub-quadratic extension |

See paper for proofs and NISQ apps.

## License
MIT License—free for research/education.

## Citation
If using, cite the accompanying paper and this implementation:

- Paper: Nasir, M. S. (2025). *Probing Quantum Dynamics Through Explainable AI: Non-Invasive Entanglement Analysis via Neural Proxies and SHAP* .  

- Software:  
  ```bibtex
  @misc{quantum_xai_app,
    title        = {Quantum XAI Mini-App: Interactive Exploration of Quantum Dynamics with Neural Networks and SHAP},
    author       = {Muhammad Shaharyar Nasir},
    year         = {2025},
    howpublished = {\url{https://github.com/ShaharyarNasir/Quantum-XAI-App}},
    note         = {Interactive Gradio-based implementation accompanying the research paper "Probing Quantum Dynamics Through Explainable AI: Non-Invasive Entanglement Analysis via Neural Proxies and SHAP" (Sep 2025)},
    month        = {Sep}
  }
