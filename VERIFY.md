# Verifying the released weights

Four checks that the 24 checkpoints in `checkpoints/synthetic` are intact,
have the architecture the manifest claims, and produce the reported
numbers. Runs on a laptop, about ten minutes.

## 1. Setup

Python 3.10 or newer.

```bash
git clone https://github.com/gavtoski/PACE_IVIM.git
cd PACE_IVIM
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "numpy<2"
```

The NumPy pin is required. PyTorch wheels are built against NumPy 1.x.

## 2. Integrity

Every checkpoint carries a SHA-256 in `configs/models.json`.

```bash
python -c "
import hashlib, json, pathlib
doc = json.load(open('configs/models.json'))
bad = sum(hashlib.sha256(pathlib.Path(e['target_rel']).read_bytes()).hexdigest()
          != e['sha256'] for e in doc['checkpoints'])
print(f'{len(doc[\"checkpoints\"])} checkpoints, {bad} bad')
"
```

Expected: `24 checkpoints, 0 bad`

## 3. Architecture

Each checkpoint should load into a network built from its manifest entry
with no key left over on either side. Strict loading compares parameter
names in both directions, so a wrong architecture raises rather than
silently dropping weights.

```bash
python -c "
from pace import common as C
M = C.load_manifest(C.load_paths())
bad = 0
for ck in M.checkpoints:
    try:
        C.load_net(ck.model, ck.snr_db, ck.seed, device='cpu')
    except Exception as e:
        bad += 1; print(f'FAIL {ck.public_name}: {type(e).__name__}')
print(f'{len(M.checkpoints)} checkpoints, {bad} failures')
"
```

Expected: `24 checkpoints, 0 failures`

Three flags change what a network computes without changing any parameter
shape, so a checkpoint loads cleanly into a wrongly configured network and
returns different numbers. Each is checkable from the weights or the
outputs, independently of the manifest.

```bash
python -c "
from pace import common as C
import torch
for m in ['dnn','pace','cnn_fusion']:
    k = torch.load(f'checkpoints/synthetic/{m}_snr25_seed19.pt',
                   map_location='cpu', weights_only=True).keys()
    fusion = ('concat'    if any('cross_attn.proj' in x for x in k) else
              'attention' if any('cross_attn.in_proj_weight' in x for x in k)
              else 'none')
    p = C.load_result(m, 25, 19); ok = p.valid()
    frac = (p.params['Fint'][ok] + p.params['Fmv'][ok]).max()
    dint = p.params['Dint'][ok].min()
    print(f'{m:<12} fusion={fusion:<10} '
          f'{\"softmax\" if frac > 0.6 else \"sigmoid\":<8} '
          f'{\"ordered\" if dint < 0.0015 else \"independent\"}')
"
```

Expected:

```
dnn          fusion=none       sigmoid  independent
pace         fusion=attention  softmax  ordered
cnn_fusion   fusion=concat     softmax  ordered
```

Why each works. **Fusion** is in the weight names: cross attention stores
`cross_attn.in_proj_weight`, concat stores `cross_attn.proj`, a signal only
model has neither. **Softmax** caps `Fint + Fmv` at 1, while the sigmoid
alternative caps them at 0.4 and 0.2, so a sum above 0.6 can only be
softmax. **Ordering** makes `Dint = Dpar + softplus(...)`, so `Dint`
inherits `Dpar`'s floor and can fall below 0.0015; without it `Dint` has a
hard floor there. Checking `Dpar < Dint < Dmv` does not distinguish them,
since the bound ranges are disjoint and that holds either way.

## 4. Behaviour

Runs each checkpoint on the test set and compares against the committed
results. Two to three minutes.

```bash
python scripts/reproduce_synthetic_results.py --compare
```

Output goes to `results/synthetic_regenerated`. The reference cannot be
overwritten.

Expected agreement, comparing the mean of each parameter:

| SNR | difference | voxels retained |
|-----|-----------|-----------------|
| 35 dB | under 0.1 % | 38,000 |
| 30 dB | under 0.2 % | 37,993 |
| 25 dB | under 0.7 % | about 37,910 |
| 20 dB | 2 to 3 %    | 37,025 |

Voxel level equality is not expected. Rician noise is injected at
inference and the random stream depends on the compute backend, so a run
on Metal or CPU retains a different subset of voxels than the original
CUDA run and sees different noise. That is why agreement degrades as SNR
falls: at 35 dB nothing is rejected and both runs evaluate the same
voxels; at 20 dB about 975 differ.

Two signs the difference is noise and not error: the retained voxel count
is identical across all three neural models at a given SNR, since
rejection happens before the network sees the data; and at 20 dB all six
parameters shift the same direction together. Above 5 percent, or one
model disagreeing while the others match, would mean something else.

## 5. Figures

```bash
python scripts/make_manuscript_figures.py
```

Reads `results/synthetic`, writes `figures/`. No GPU or PyTorch needed.
The ranking printed at the end should match exactly, since nothing here is
random:

```
1.  CNN Fusion  0.01500 +/- 0.00656
2.  PACE        0.01542 +/- 0.00628
3.  DNN         0.01671 +/- 0.00542
4.  LSQ         0.02058 +/- 0.00555
5.  MAP         0.02118 +/- 0.00586
6.  NNLS        0.02814 +/- 0.00754
```

## Not included

The in-vivo cohort is protected health information and cannot be shared,
so the in-vivo code and the brain map experiments are outside this
repository. The training set is also excluded; it is needed only to
retrain, and is available on request. See the Retraining section of
README.md for checking a retrained network against these weights.

## Problems

Open an issue at https://github.com/gavtoski/PACE_IVIM/issues with the
command, the output, and:

```bash
python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"
```
