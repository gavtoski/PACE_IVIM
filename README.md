# PACE-IVIM

**Physics-anchored Cross-attention Embedding (PACE)** is a multimodal deep learning framework for three-compartment IVIM parameter estimation. PACE integrates co-registered T1w and FLAIR spatial information with a physics-informed neural network through gated cross-attention. An additional anatomy-weighted spatial prior further reduces parameter RMSE with some distribution trade-offs.

**Preprint (under peer review):** Hoang N, Faiyaz A, Mensah EA, Uddin MN, Zhong J, Schifitto G. *Improved three component IVIM fitting using physics-informed neural network with multimodal spatial prior.* arXiv:2610.28873 (2026). [Read preprint](https://arxiv.org/abs/2610.28873).

## Quick Start

Requires Python 3.10+.

```bash
git clone https://github.com/gavtoski/PACE_IVIM.git
cd PACE_IVIM
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "numpy<2"
python run_PACE_IVIM.py
```

### Default Run

Running `python run_PACE_IVIM.py` automatically:

1. **Downloads data:** Retrieves missing synthetic test data and pretrained weights from the manuscript's dataset (Hoang et al., 2026) from [Zenodo](https://doi.org/10.5281/zenodo.23746536). This should recreate the result in Hoang et al., 2026. No training is required, unless retraining is needed to apply PACE to a different cohort. Please see the section below for training options.
2. **Runs inference:** Evaluates four neural networks (PACE, PACE with spatial prior, DNN, IVIM3C) at SNRs of 20, 25, 30, and 35 dB, with five seeds at 25 dB.
3. **Loads conventional fits:** Uses the included LSQ, NNLS, and MAP results.
4. **Evaluates performance:** Computes signal RMSE, parameter RMSE, and lesion CNR RMSE across all seven models.
5. **Generates outputs:** Saves summary metrics (signal RMSE, parameter RMSE, and lesion CNR RMSE) to `results/summary.csv` and detailed metrics to `results/report.txt`. Comparison figures are saved to `figures/model_comparison/`.

Optional architecture ablations are also included (`--ablations`) for exploratory use only.

### Repository Structure

```text
PACE_IVIM/
├── run_PACE_IVIM.py    Main runner
├── configs/            Model and ablation settings
├── pace/               Network architecture and fitting methods
├── scripts/            Training, inference, and evaluation
├── data/synthetic/     Synthetic data and b-values
├── checkpoints/        Pretrained model weights
├── results/            Inference outputs and metrics
└── figures/            Comparison and ablation figures (PNG/TIFF)
```

Data, checkpoint, result, and figure directories are populated during execution.

## Training

The default run uses released test data and weights. To retrain the networks, download the training and validation datasets (~1.7 GB), then train with `--train`. The training data creation procedure is included in the manuscript (Hoang et al., 2026).

```bash
python run_PACE_IVIM.py --download-only training               # Training and validation data
python run_PACE_IVIM.py --train --models PACE --epochs 1        # Quick check, one epoch
python run_PACE_IVIM.py --train --models PACE PACE_prior DNN    # Retrain selected models (35 epochs)
python run_PACE_IVIM.py --train --ablations                     # Also retrain architecture ablations
python run_PACE_IVIM.py --train --overwrite                     # Replace earlier retrained weights
```

For training on data stored elsewhere for different cohorts:

```bash
python run_PACE_IVIM.py --train --models PACE --data-dir /path/to/data
```

Retrained weights are saved to `checkpoints/retrained/` and their metrics to `results/retrained/`; the released weights and results are not modified. Existing retrained weights are reused unless `--overwrite` is given, so add it after a quick check. Retraining all four networks at every SNR (32 runs) is best done on a GPU.

## Common Commands

```bash
python run_PACE_IVIM.py --list                         # Available models and ablations
python run_PACE_IVIM.py --models PACE DNN --snrs 25    # Compare selected models
python run_PACE_IVIM.py --ablations                    # Include architecture ablations
python run_PACE_IVIM.py --conventional                 # Refit LSQ, NNLS, MAP
python run_PACE_IVIM.py --skip-inference               # Regenerate metrics and figures
python run_PACE_IVIM.py --help                         # All options
```

Available models: `PACE`, `PACE_prior`, `DNN`, `IVIM3C`, `LSQ`, `NNLS`, `MAP`. Model and ablation configurations are defined in `configs/arms.json`.

## Training and Test Data

Synthetic data and pretrained weights are available on [Zenodo](https://doi.org/10.5281/zenodo.23746536) https://doi.org/10.5281/zenodo.23746536. Training and validation data are downloaded as described in [Training](#training).

Data can be stored elsewhere using `--data-dir /path/to/data`.

The in vivo clinical data used in the manuscript are not publicly available.

## Prior Physics-Informed Network Architectures

PACE builds on the three-compartment physics-informed IVIM network of Voorter et al. [1] and the spatial deep learning framework of Kaandorp et al. [2]. The original Voorter network is included as the IVIM3C comparator in `pace/PaulienVoorter_deep.py`, with attribution preserved.

## Citation

If you find PACE-IVIM helpful, please cite the **arXiv preprint** (publication pending):

> Hoang N, Faiyaz A, Mensah EA, Uddin MN, Zhong J, Schifitto G. Improved three component IVIM fitting using physics-informed neural network with multimodal spatial prior. *arXiv preprint* arXiv:2610.28873; 2026. [https://arxiv.org/abs/2610.28873](https://arxiv.org/abs/2610.28873)

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

For the synthetic data and pretrained weights, please also cite:

> Hoang N, Faiyaz A, Mensah EA, Uddin MN, Zhong J, Schifitto G. *Synthetic data and trained weights for PACE-IVIM.* Zenodo; 2026. [doi:10.5281/zenodo.23746536](https://doi.org/10.5281/zenodo.23746536)

## References

1. Voorter PHM, Backes WH, Gurney-Champion OJ, et al. Improving microstructural integrity, interstitial fluid, and blood microcirculation images from multi-b-value diffusion MRI using physics-informed neural networks in cerebrovascular disease. *Magnetic Resonance in Medicine*. 2023;90(4):1657–1671. [doi:10.1002/mrm.29753](https://doi.org/10.1002/mrm.29753). [Code](https://github.com/paulienvoorter/IVIM3brain-NET).
2. Kaandorp MPT, Zijlstra F, Karimi D, Gholipour A, While PT. Incorporating spatial information in deep learning parameter estimation with application to the intravoxel incoherent motion model in diffusion-weighted MRI. *Medical Image Analysis*. 2025;101:103414. [doi:10.1016/j.media.2024.103414](https://doi.org/10.1016/j.media.2024.103414). [Code](https://github.com/Mishakaandorp/Incorporating_spatial_information_in_deep_learning_parameter_estimation).

## Acknowledgments

Supported by the National Institutes of Health (R01AG077156).

## License

MIT License.
