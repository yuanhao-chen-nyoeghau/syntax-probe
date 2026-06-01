# syntax_probe

Probing large language models for formal syntactic structure.

Authors:
- Yuanhao Chen
- Peter Chin

Paper: 
- *Probing Minimalist Phase Structure in LLMs: What Universal Dependencies Cannot Represent*
- <https://arxiv.org/abs/2605.26431>

```bibtex
@misc{chenProbingMinimalistPhase2026,
  title = {Probing {{Minimalist Phase Structure}} in {{LLMs}}: {{What Universal Dependencies Cannot Represent}}},
  shorttitle = {Probing {{Minimalist Phase Structure}} in {{LLMs}}},
  author = {Chen, Yuanhao and Chin, Peter},
  year = 2026,
  month = may,
  number = {arXiv:2605.26431},
  eprint = {2605.26431},
  primaryclass = {cs.CL},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2605.26431},
  archiveprefix = {arXiv},
}
```

## What this codebase does

This codebase implements a pipeline for testing whether LLM hidden states encode formal syntactic structure beyond what dependency parses capture. The methodology, generalized from Kennedy (2025), is:

1. Train a Hewitt & Manning–style structural probe on gold dependency parses from an out-of-domain corpus (UD English EWT).
2. Apply the trained probe to minimal-pair stimuli where dependency parses are identical across conditions but generative-syntactic structure differs.
3. Test whether the probe's predicted distances vary across conditions in directions predicted by formal syntactic theory.

If conditions differ in probe output despite having identical dependency targets, this is evidence that the LLM encodes structural information beyond dependency distance.

## Quick start

```bash
# Install with uv. This creates the project venv, registers all console
# scripts (syntax-probe, wh-stats, cc-stats, run-all-stats, make-figures),
# and pulls in the analysis-only deps.
uv pip install -e ".[dev,analysis]"

# Download UD English EWT
uv run syntax-probe download-corpus --output data/ud_ewt

# Install spaCy's transformer parser (used by the verify-stimuli recipes for
# best parsing accuracy on object-relative-clause and other tricky stimuli).
# ~430 MB download; uses GPU when --set experiment.gpu_id=N is passed.
uv run python -m spacy download en_core_web_trf

# Run the parser-distance verification check (Task 0.5).
uv run syntax-probe run configs/verify_stimuli/wh_extraction_spacy.yaml --set experiment.gpu_id=0
uv run syntax-probe run configs/verify_stimuli/c_command_spacy.yaml     --set experiment.gpu_id=0

# Train probes on UD EWT (Qwen2.5-1.5B is the default; ungated)
uv run syntax-probe run configs/probe_training/qwen25_1_5b_ud_ewt.yaml

# Apply trained probes to experimental stimuli. The apply-probes config
# auto-resolves to the latest matching probe-training run, so you don't need
# to edit the run id by hand. To apply a specific (non-latest) run:
#   uv run syntax-probe run configs/applications/wh_extraction_qwen25_1_5b.yaml \
#     --set experiment.probe_run_dir=results/probe_training/<run-id>
uv run syntax-probe run configs/applications/wh_extraction_qwen25_1_5b.yaml
uv run syntax-probe run configs/applications/c_command_qwen25_1_5b.yaml  # for c-command

# Run several configs back-to-back. Globs are handled by the shell, and
# --set overrides apply to every config in the batch:
uv run syntax-probe run configs/applications/wh_extraction_*.yaml
uv run syntax-probe run configs/probe_training/{llama32_1b,qwen3_8b,gemma3_12b}_ud_ewt.yaml

# Compute statistics on the apply-probes outputs. Single-model mode picks the
# most recent run; multi-model mode iterates over the figures registry.
uv run wh-stats                              # single model, newest run
uv run wh-stats --all-models                 # every registered model
uv run run-all-stats --all-models            # both experiments, all models

# Generate paper figures (default: the registry's default=True models).
uv run make-figures --all-models             # every model with runs
```

## Methodology references

- Hewitt, J. & Manning, C. D. (2019). *A Structural Probe for Finding Syntax in Word Representations.* NAACL.
- Kennedy, M. K. (2025). *Evidence of Generative Syntax in Large Language Models.* CoNLL.
