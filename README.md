# PACE-IVIM

**PACE** estimates three-compartment IVIM parameters from diffusion MRI using a physics-informed neural network with gated cross-attention to T1-weighted and FLAIR images. An optional anatomy-weighted **spatial prior** uses neighboring diffusion signals during training.

**Preprint (under peer review):** Hoang N, Faiyaz A, Mensah EA, Uddin MN, Zhong J, Schifitto G. *Improved three component IVIM fitting using physics-informed neural network with multimodal spatial prior.* **arXiv preprint**, arXiv:2610.28873 (2026). [Read the preprint](https://arxiv.org/abs/2610.28873).

## Quick Start

Requires Python 3.10+.

```bash
git clone https://github.com/gavtoski/PACE_IVIM.git
cd PACE_IVIM
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "numpy<2"
python run_PACE_IVIM.py
```

**By default**, the runner downloads test data and pretrained weights from Zenodo **only if missing**, then performs inference and generates metrics and figures. Existing files are reused on later runs; `--no-download` disables automatic downloads.

## Common Commands

```bash
python run_PACE_IVIM.py --list                           # Available models and ablations
python run_PACE_IVIM.py --models PACE DNN --snrs 25      # Selected models at 25 dB
python run_PACE_IVIM.py --ablations                      # Architecture ablations
python run_PACE_IVIM.py --skip-inference                 # Recompute reports and figures
python run_PACE_IVIM.py --download-only training         # Training/validation data
```

Models: **PACE**, **PACE_prior**, **DNN**, **IVIM3C**, **LSQ**, **NNLS**, and **MAP**. All model and ablation settings are in `configs/arms.json`; see `--help` for additional options.

## Outputs

- `results/summary.csv` — signal/parameter RMSE and lesion CNR RMSE
- `results/report.txt` — detailed evaluation metrics
- `figures/` — comparison and ablation figures

## Training and Test Data

[Synthetic datasets and pretrained weights](https://doi.org/10.5281/zenodo.23746536) are hosted on Zenodo. Test data and model weights (~56 MB) are fetched automatically when missing; training/validation data (~1.7 GB) are optional. Training scripts are in `scripts/`. Clinical in-vivo data are not distributed.

## Prior Physics-Informed Net Architecture

PACE builds on the three-compartment IVIM network of Voorter et al. [1] and the spatial attention approach of Kaandorp et al. [2]. Voorter's network is included as the IVIM3C comparator.

## Citation — arXiv Preprint

Please cite the **2026 arXiv preprint (not yet a peer-reviewed journal article)**:

```bibtex
@misc{hoang2026improvedcomponentivimfitting,
  title         = {Improved three component IVIM fitting using physics-informed neural network with multimodal spatial prior},
  author        = {Nhat Hoang and Abrar Faiyaz and Emmanuel A. Mensah and Md Nasir Uddin and Jianhui Zhong and Giovanni Schifitto},
  year          = {2026},
  eprint        = {2610.28873},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url           = {https://arxiv.org/abs/2610.28873}
}
```

If you use the data or weights, please also cite the Zenodo record:

> Hoang N, Faiyaz A, Mensah EA, Uddin MN, Zhong J, Schifitto G. Synthetic data and trained weights for PACE-IVIM. Zenodo; 2026. doi: [10.5281/zenodo.23746536](https://doi.org/10.5281/zenodo.23746536).

## References

1. Voorter PHM, Backes WH, Gurney-Champion OJ, Wong S-M, Staals J, van Oostenbrugge RJ, van der Thiel MM, Jansen JFA, Drenthen GS. Improving microstructural integrity, interstitial fluid, and blood microcirculation images from multi-b-value diffusion MRI using physics-informed neural networks in cerebrovascular disease. *Magnetic Resonance in Medicine*. 2023;90(4):1657-71. doi: [10.1002/mrm.29753](https://doi.org/10.1002/mrm.29753).
2. Kaandorp MPT, Zijlstra F, Karimi D, Gholipour A, While PT. Incorporating spatial information in deep learning parameter estimation with application to the intravoxel incoherent motion model in diffusion-weighted MRI. *Medical Image Analysis*. 2025;101:103414. doi: [10.1016/j.media.2024.103414](https://doi.org/10.1016/j.media.2024.103414).

Supported by NIH R01AG077156.

## License

MIT.
