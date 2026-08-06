# Harmonic Representations in Symbolic Music Transformers are Localized and Controllable

**Adi Silberschein**<sup>1</sup> &nbsp; **Megan Wei**<sup>2</sup> &nbsp; **Tamar Rott Shaham**<sup>3</sup>

<sup>1</sup>Weizmann Institute of Science &nbsp;&nbsp; <sup>2</sup>Brown University &nbsp;&nbsp; <sup>3</sup>MIT CSAIL

[[Paper]](https://arxiv.org/abs/XXXX.XXXXX) &nbsp; [[Website]](https://adisilber.github.io/harmonic-interp/)

---

## Overview

We study how harmonic structure is internally represented in Aria, a LLaMA-based symbolic music transformer, using Bach chorales as a controlled testbed.

Using **activation patching**, we find that harmonic decisions are causally localized progressively from middle layers onward: the causal effect emerges around layer 6 and grows through later layers. **Linear probes** show that mode (major/minor) becomes decodable in the same layer range, while key identity emerges later in the network. Finally, **steering vectors** derived from probe directions reliably redirect the model's cadence resolutions — inducing deceptive resolutions — while preserving structural coherence and generation quality.

## Repository structure

```
reproduce/             # Self-contained reproduction pipeline (+ run_the_experiment.ipynb)
config/                # Aria model configs; model weights are downloaded, not committed
docs/                  # Paper website (served via GitHub Pages)
```

The Aria model code is installed as a dependency ([EleutherAI/aria](https://github.com/EleutherAI/aria)) rather than vendored into this repo.

## Setup

```bash
# Clone the repo
git clone https://github.com/adiSilber/harmonic-interp.git
cd harmonic-interp

# Install dependencies (also pulls EleutherAI/aria and ariautils from GitHub)
pip install -e .
```

**Model weights:** run `python reproduce/download_model.py` — fetches the Aria checkpoint
(`model-gen.safetensors`, ~2.6 GB) from HuggingFace [`loubb/aria-medium-base`](https://huggingface.co/loubb/aria-medium-base)
into `config/models/aria-medium-base/` (where the pipeline expects it). Skips the download if
already present.

**Data:** the JSB Chorales are downloaded automatically by the notebook's Step 1
(`python reproduce/download_jsb_chorales.py`) into `reproduce_output/jsb_chorales_midi/`.

## Reproducing the results

Everything runs from **`reproduce/run_the_experiment.ipynb`**, which walks through the full pipeline end to end (model + data download, activation extraction, probing, patching, steering, and quality metrics). The individual stages live in `reproduce/`:

### 1. Activation patching
`extract_patching_activations.py` + `run_per_layer_patching.py` — run factual/counterfactual pairs through Aria and patch the residual stream one layer at a time.

### 2. Linear probing
`extract_probe_activations.py`, `prepare_probe_data.py`, `train_probes.py` — extract residual-stream activations and train linear probes for mode and key identity at each layer.

### 3. Steering
`build_steering_directions.py` + `run_steering.py` — derive steering vectors from probe directions and apply them during generation; `label_resolutions.py`, `compute_quality_metrics.py`, and `compute_fmd.py` score the outputs.

## Citation

```bibtex
@misc{silberschein2025harmonic,
  title={Harmonic Representations in Symbolic Music Transformers are Localized and Controllable},
  author={Silberschein, Adi and Wei, Megan and Rott Shaham, Tamar},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## Acknowledgements

This work builds on [EleutherAI/aria](https://github.com/EleutherAI/aria) by Bradshaw et al. (ISMIR 2025).
