# Verifying the released results

Checks that the trained networks in this repository produce the numbers
reported in the manuscript. 

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

## 2. Check the weights

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

## 3. Check the architecture

Every checkpoint should load into a network built from its manifest entry
with nothing left over. Strict loading compares parameter names both ways,
so a wrong architecture raises rather than silently dropping weights.

```bash
python -c "
from pace import common as C
M = C.load_manifest(C.load_paths())
bad = 0
for ck in M.checkpoints:
    try:
        net, _ = C.load_net(ck.model, ck.snr_db, ck.seed, device='cpu')
    except Exception as e:
        bad += 1; print(f'FAIL {ck.public_name}: {type(e).__name__}')
print(f'{len(M.checkpoints)} checkpoints, {bad} failures')
"
```

Expected: `24 checkpoints, 0 failures`

## 4. Reproduce the results

Loads each checkpoint, runs it on the test set, and compares against
`results/synthetic`. Takes about two to three minutes.

```bash
python scripts/reproduce_synthetic_results.py --compare
```

Output goes to `results/synthetic_regenerated`. The reference cannot be
overwritten.

**Expected agreement**, comparing the mean of each parameter:

| SNR | difference | voxels retained |
|-----|-----------|-----------------|
| 35 dB | under 0.1 % | 38,000 |
| 30 dB | under 0.2 % | 37,993 |
| 25 dB | under 0.7 % | about 37,910 |
| 20 dB | 2 to 3 %    | 37,025 |

Voxel-level equality is not expected due to injected Rician noise, but the output
results should be close to the reference.

## 5. Regenerate the figures

```bash
python scripts/make_manuscript_figures.py
```

Reads `results/synthetic`, writes `figures/`. No GPU or PyTorch needed.

| Figure | Content |
|--------|---------|
| 1   | Signal RMSE against SNR, across all b-value range |
| 2.0 | Parameter RMSE against SNR |
| 2.1 | Parameter RMSE, bar plot |
| 2.2 | Bland-Altman against ground truth |
| 3.1 | WMH against NAWM lesion CNR |
| 4.1 | Learned spatial gate values |

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
retrain, and is available on request.

