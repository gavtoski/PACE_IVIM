# PACE

**Prior-Informed Anatomical Context Encoding for 3-Compartment IVIM MRI**

Provides reference's implementation and synthetic evaluation for the PACE model, which
fits a three-component IVIM model to diffusion-weighted brain MRI:

```
S(b) = S0 * [ Fpar*exp(-b*Dpar) + Fint*exp(-b*Dint) + Fmv*exp(-b*Dmv) ]
```

Six parameters per voxel: Dpar, Fint, Dint, Fmv, Dmv, S0, with
Fpar = 1 - Fint - Fmv. The three compartments correspond to parenchymal
tissue, interstitial fluid, and microvascular blood.

PACE conditions parameter estimation on anatomical context by fusing the
IVIM signal encoding with T1, FLAIR, and b=0 spatial tokens through
cross-attention, under a softmax simplex constraint on the compartment
fractions and a monotonic ordering constraint on the diffusion coefficients.

## Status

This repository currently contains the **synthetic evaluation** only.
The in vivo code and the synthetic brain map experiments are not included.

Everything needed to check the synthetic results is here: the trained
weights, the test data, the inference outputs, and the code that turns one
into the next. **[VERIFY.md](VERIFY.md)** walks through confirming that the
released checkpoints reproduce the published numbers, in about ten minutes
on a laptop.

The complete repository will be released once the associated manuscript
has been accepted.

## Models

| Name | Description |
|------|-------------|
| `pace` | Cross-attention fusion with softmax simplex constraint |
| `cnn_fusion` | Concatenation fusion with softmax simplex constraint |
| `dnn` | Signal only baseline, no spatial conditioning |
| `lsq` | Two step least squares |
| `nnls` | Non-negative least squares spectral decomposition |
| `map` | Bayesian maximum a posteriori |

The three neural models ship as trained weights under `checkpoints/synthetic/`.
The three conventional methods are fit per voxel at run time and have no
weights. `configs/models.json` maps each public name to its architecture
configuration and provenance.

## Layout

```
pace/          library code shared by the scripts
scripts/       entry points
configs/       model manifest and path configuration
checkpoints/   trained weights
data/          synthetic test and validation sets
results/       precomputed inference outputs
figures/       generated figures
```

## Installation

Python 3.10 or newer.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "numpy<2"
cp configs/paths.example.yaml configs/paths.yaml
```

The NumPy pin is required: current PyTorch wheels are built against
NumPy 1.x and fail on conversion to arrays under NumPy 2.x.

## Usage

Redraw every figure from the committed inference results. Needs neither a
GPU nor PyTorch:

```bash
python scripts/make_manuscript_figures.py
python scripts/make_manuscript_figures.py --figures 1 2.0
python scripts/make_manuscript_figures.py --snrs 25 30 35
```

Six figures are produced: signal RMSE (1), per parameter RMSE as lines
(2.0) and as bars (2.1), Bland Altman against ground truth (2.2), lesion
CNR (3.1), and the learned spatial gate values (4.1).

Re-run the trained networks over the test set and compare the output
against the committed results:

```bash
python scripts/reproduce_synthetic_results.py --compare
python scripts/reproduce_synthetic_results.py --models pace --snrs 25
```

The two form a chain:

```
checkpoints  ->  reproduce_synthetic_results.py  ->  results
results      ->  make_manuscript_figures.py      ->  figures
```

See **[VERIFY.md](VERIFY.md)** for what agreement to expect and why the
comparison is statistical rather than exact.

## Data availability

The synthetic dataset, trained weights and inference outputs required to
reproduce every figure in this repository are included here. The synthetic
training set is not: it is needed only to retrain from scratch, and is
available from the authors on request.

The in-vivo NeuroCovid cohort cannot be shared. It is protected health
information under an institutional review board protocol that does not
permit redistribution. The in-vivo training and evaluation code is
available from the authors on reasonable request.

## Prior work

PACE builds on an existing line of deep learning approaches to IVIM
parameter estimation. The relevant lineage, in order:

**Barbieri et al. (2020)** introduced unsupervised deep learning for IVIM
model fitting, training a network to reconstruct the measured signal rather
than to regress against least squares targets [1].
Source: https://github.com/sebbarb/deep_ivim

**Kaandorp et al. (2021)** improved that formulation into the physics
informed IVIMNET, addressing training stability and parameter
constraints [2].
Source: https://github.com/oliverchampion/IVIMNET

**Voorter et al. (2023)** extended physics informed networks to the three
component cerebral IVIM model, separating parenchymal diffusion,
interstitial fluid, and microvascular pseudo-diffusion in cerebrovascular
disease [3]. The parameter bounds used in this work follow their
Table S1. Note that the repository below carries the working title
"Physics-informed neural networks improve three-component model fitting of
intravoxel incoherent motion MR imaging in cerebrovascular disease"; the
paper appeared under the published title given in [3].
Source: https://github.com/paulienvoorter/IVIM3brain-NET

**Kaandorp et al. (2025)** demonstrated that attention based architectures
can incorporate spatial information from neighbouring voxels, trained on
synthetic data with spatial correlation structure [4]. A subsequent
comparative study evaluated conventional, Bayesian, and spatially aware
deep learning fitting in glioma grading [5].
Source: https://github.com/Mishakaandorp/Incorporating_spatial_information_in_deep_learning_parameter_estimation

PACE differs from [4] in the source of spatial information. Rather than
learning correlations among neighbouring IVIM signals, it conditions on
co-registered anatomical contrast, T1 and FLAIR, through per parameter
gated cross attention.

## References

[1] S. Barbieri, O. J. Gurney-Champion, R. Klaassen, and H. C. Thoeny.
"Deep learning how to fit an intravoxel incoherent motion model to
diffusion-weighted MRI." *Magnetic Resonance in Medicine*, 83(1):312-321,
2020. doi:10.1002/mrm.27910

[2] M. P. T. Kaandorp, S. Barbieri, R. Klaassen, H. W. M. van Laarhoven,
H. Crezee, P. T. While, A. J. Nederveen, and O. J. Gurney-Champion.
"Improved unsupervised physics-informed deep learning for intravoxel
incoherent motion modeling and evaluation in pancreatic cancer patients."
*Magnetic Resonance in Medicine*, 86(4):2250-2265, 2021.
doi:10.1002/mrm.28852

[3] P. H. M. Voorter, W. H. Backes, O. J. Gurney-Champion, S.-M. Wong,
J. Staals, R. J. van Oostenbrugge, M. M. van der Thiel, J. F. A. Jansen,
and G. S. Drenthen. "Improving microstructural integrity, interstitial
fluid, and blood microcirculation images from multi-b-value diffusion MRI
using physics-informed neural networks in cerebrovascular disease."
*Magnetic Resonance in Medicine*, 2023. doi:10.1002/mrm.29753

[4] M. P. T. Kaandorp, F. Zijlstra, D. Karimi, A. Gholipour, and
P. T. While. "Incorporating spatial information in deep learning parameter
estimation with application to the intravoxel incoherent motion model in
diffusion-weighted MRI." *Medical Image Analysis*, 101:103414, 2025.
doi:10.1016/j.media.2024.103414

[5] M. P. T. Kaandorp et al. "A Comparative Study of IVIM-MRI Fitting
Techniques in Glioma Grading: Conventional, Bayesian, and Voxel-Wise and
Spatially-Aware Deep Learning Approaches." *Journal of Magnetic Resonance
Imaging*, 64(1), 2026.

## Citation

A citation will be added once the associated manuscript is published. Until
then, please cite this repository by URL.

## License

MIT. See [LICENSE](LICENSE).

`pace/conventional_models.py` derives from the IVIMNET project by
Gurney-Champion and Kaandorp [2], which is also MIT licensed.
