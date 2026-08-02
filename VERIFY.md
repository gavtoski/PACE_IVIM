# Verifying the released results

This guide walks through checking that the trained networks in this
repository actually produce the numbers reported in the manuscript. Every
step runs on a laptop. No GPU is required, and nothing here needs data that
is not included.

There are two claims to check, and they are separate:

1. The figures follow from the inference results shipped in `results/`.
2. Those inference results follow from the checkpoints shipped in
   `checkpoints/`.

The second is the stronger claim, and section 4 is where it is tested.
Section 5 covers the first. Sections 2, 3 and 6 confirm that the released
weights are intact and that their architecture is what the manifest says
it is.

Expect the whole guide to take about ten minutes.

---

## 1. Environment

Python 3.10 or newer is required. The code uses union type syntax and
`sys.stdlib_module_names`, neither of which exists in 3.9.

```bash
git clone https://github.com/gavtoski/PACE_IVIM.git
cd PACE_IVIM

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "numpy<2"
```

The NumPy pin is deliberate. Current PyTorch wheels are compiled against
NumPy 1.x, and pairing them with NumPy 2.x produces a `_ARRAY_API not
found` failure the moment a tensor is converted to an array.

Confirm the install:

```bash
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

---

## 2. Check the released assets

Every checkpoint carries a SHA-256 in `configs/models.json`. Confirming
them proves the weights you have are the weights that were released.

```bash
python -c "
import hashlib, json, pathlib
doc = json.load(open('configs/models.json'))
bad = 0
for e in doc['checkpoints']:
    p = pathlib.Path(e['target_rel'])
    if not p.is_file():
        print(f'MISSING  {p}'); bad += 1; continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != e['sha256']:
        print(f'MISMATCH {p}'); bad += 1
print(f'{len(doc[\"checkpoints\"])} checkpoints, {bad} problem(s)')
"
```

Expect **0 problems** across 24 checkpoints.

The architecture of each network is also readable directly from its
weights, without trusting any configuration file. Cross attention stores
`cross_attn.in_proj_weight`; concatenation fusion stores `cross_attn.proj`;
a signal only model has neither.

```bash
python -c "
import torch
for m in ['dnn','pace','cnn_fusion']:
    k = torch.load(f'checkpoints/synthetic/{m}_snr25_seed19.pt',
                   map_location='cpu', weights_only=True).keys()
    mode = ('concat'    if any('cross_attn.proj' in x for x in k) else
            'attention' if any('cross_attn.in_proj_weight' in x for x in k)
            else 'none')
    n = sum(v.numel() for v in torch.load(
        f'checkpoints/synthetic/{m}_snr25_seed19.pt',
        map_location='cpu', weights_only=True).values())
    print(f'{m:<12} fusion={mode:<10} {n:>9,} values')
"
```

Expected:

```
dnn          fusion=none            7,303 values
pace         fusion=attention     194,802 values
cnn_fusion   fusion=concat        198,866 values
```

`dnn` reporting `none` is correct: that model has no spatial pathway, so it
has no fusion block at all.

Note that `cnn_fusion` has slightly **more** parameters than `pace`. The
two differ only in the fusion module, and concatenation fusion is the
larger of the two, so the comparison between them is not confounded by
model capacity.

---

## 3. Check that the package loads

Every released checkpoint should load into a network built from its
manifest entry, with no key left over on either side. Strict loading is
what makes this meaningful: it compares the full set of parameter names in
both directions, so any error in the reconstructed architecture raises
instead of silently dropping weights.

```bash
python -c "
from pace import common as C
M = C.load_manifest(C.load_paths())
bad = 0
for ck in M.checkpoints:
    try:
        net, _ = C.load_net(ck.model, ck.snr_db, ck.seed, device='cpu')
        n = sum(p.numel() for p in net.parameters())
        print(f'{ck.public_name:<26} {n:>9,} params  ok')
    except Exception as e:
        bad += 1
        print(f'{ck.public_name:<26} FAIL  {type(e).__name__}')
print(f'--- {bad} failures ---')
"
```

Expect **0 failures**. Parameter counts are 7,245 for `dnn`, 194,744 for
`pace` and 198,808 for `cnn_fusion`. These are 58 lower than the value
counts in section 2 because `net.parameters()` excludes registered
buffers: the 46 b-values and the two six element bound vectors.

---

## 4. Reproduce the inference results

This is the substantive check. It loads each released checkpoint, runs it
over the synthetic test set, and compares the output against the results
committed to `results/synthetic`.

```bash
python scripts/reproduce_synthetic_results.py --compare
```

Output is written to `results/synthetic_regenerated`. The script refuses to
write into `results/synthetic`, so the reference cannot be overwritten.

On a laptop this takes two to three minutes for all 24 checkpoints.

### What agreement to expect

Voxel level equality is **not** expected, and its absence is not a defect.
Rician noise is injected during the forward pass, seeded through
`torch.manual_seed`. The resulting random stream depends on the compute
backend, so a run on Metal or CPU does not reproduce the original CUDA run
sample for sample. Two consequences follow:

- A different subset of voxels survives the signal cleaning step, which
  rejects voxels whose noisy signal is unusable.
- The voxels that do survive see different noise.

What should agree is the distribution. Comparing the mean of each parameter
between the reference and the regenerated output:

| SNR | relative difference | voxels retained |
|-----|--------------------|-----------------|
| 35 dB | 0.00 to 0.08 % | 38,000 of 38,000 |
| 30 dB | 0.01 to 0.13 % | 37,993 |
| 25 dB | 0.01 to 0.61 % | about 37,910 |
| 20 dB | 2.0 to 3.2 %   | 37,025 |

The trend is the mechanism made visible. At 35 dB no voxels are rejected,
both runs evaluate the same population, and the means agree to four decimal
places. As noise increases more voxels are rejected, the two retained
populations diverge, and the difference in the means grows with them.

Two further details worth noticing, both of which indicate that the
difference is the noise stream rather than an error:

- At a given SNR and seed, the retained voxel count is **identical across
  all three neural models**. Rejection happens before the network sees the
  data, so it depends only on the noise, not the architecture.
- At 20 dB, all six parameters shift in the same direction by a similar
  amount. That is a systematic difference in which voxels were retained,
  not random disagreement.

A relative difference above roughly 5 percent, or one model disagreeing
while the others match, would indicate something other than noise.

---

## 5. Regenerate the figures

```bash
python scripts/make_manuscript_figures.py
```

This reads `results/synthetic` and writes to `figures/`. It needs neither
a GPU nor PyTorch, taking about a minute.

Six figures are produced:

| Figure | Content |
|--------|---------|
| 1   | Signal RMSE against SNR, ranked mean, residual against b-value |
| 2.0 | Per parameter RMSE against SNR, line plots |
| 2.1 | Per parameter RMSE, bar grid |
| 2.2 | Bland Altman against ground truth, SNR 25 dB |
| 3.1 | WMH against NAWM lesion CNR |
| 4.1 | Learned spatial gate values, PACE against CNN Fusion |

Individual figures can be selected:

```bash
python scripts/make_manuscript_figures.py --figures 1 2.0
```

Figures 2.0 and 2.1 both report per parameter RMSE and are expected to
disagree slightly. They aggregate differently: 2.0 pools tissues and seeds
in a single average, while 2.1 collapses tissues within each (SNR, seed)
pair before averaging over those pairs. Both match the corresponding
figure in the manuscript. The difference is a property of the two
aggregations, not an error in either.

Figure 4.1 reads from a different source to the rest. Instead of the
inference results it loads the per epoch gate trajectories recorded during
training, under `checkpoints/synthetic/logs/`, so it reflects what the
networks learned rather than anything recomputed here. The gate is the
mixing weight in

```
z_X = (1 - sigma(g_X)) * z_ivim + sigma(g_X) * z_fused
```

so a value near 0 means that parameter ignores anatomical context and a
value near 1 means it depends on it. PACE and CNN Fusion share this
equation exactly and differ only in the fusion module that produces
`z_fused`, which is what makes the comparison meaningful.

The ranking printed at the end should reproduce exactly, since this step
involves no randomness:

```
  Signal RMSE ranking (SNR [20, 25, 30, 35])
  1.     CNN Fusion  RMSE = 0.01500 +/- 0.00656  (n=8)
  2.           PACE  RMSE = 0.01542 +/- 0.00628  (n=8)
  3.            DNN  RMSE = 0.01671 +/- 0.00542  (n=8)
  4.            LSQ  RMSE = 0.02058 +/- 0.00555  (n=8)
  5.            MAP  RMSE = 0.02118 +/- 0.00586  (n=8)
  6.           NNLS  RMSE = 0.02814 +/- 0.00754  (n=8)
```

Any difference here, however small, would mean the figure code is reading
something other than the committed results.

To draw the figures from your own regenerated results instead of the
reference, point `configs/paths.yaml` at that directory:

```yaml
results:
  synthetic: results/synthetic_regenerated
```

Copy `configs/paths.example.yaml` to `configs/paths.yaml` first if you have
not already. Note that at 20 dB the two sets are not directly comparable:
the reference stores all 38,000 voxels with a validity mask, whereas the
regenerated files store only the voxels that survived cleaning.

---

## 6. Inspect the architecture flags directly

Three flags determine what a network computes without changing the shape of
any parameter, which means a checkpoint will load successfully into a
network configured wrongly and quietly return different numbers. They are
therefore recorded per checkpoint in `configs/models.json` rather than
inferred at run time, and each one can be checked independently.

```bash
python -c "
from pace import common as C
M = C.load_manifest(C.load_paths())
for ck in M.checkpoints:
    if ck.seed == 19 and ck.snr_db == 25:
        print(f'{ck.model:<12} fusion={ck.fusion_mode:<10} '
              f'ordered={ck.use_ordered_diffusion} '
              f'softmax={ck.use_softmax_fractions}')
"
```

**`fusion_mode`** is visible in the weights. Cross attention stores
`cross_attn.in_proj_weight`; concatenation fusion stores `cross_attn.proj`.

**`use_softmax_fractions`** is visible twice over. It adds an `Fpar_head`
layer to the state dict, and it changes the range of the output: under the
independent sigmoid parameterisation `Fint <= 0.4` and `Fmv <= 0.2`, so
their sum cannot exceed 0.6, whereas the softmax simplex is bounded only by
1.

```bash
python -c "
from pace import common as C
for m in ['dnn','pace','cnn_fusion']:
    p = C.load_result(m, 25, 19); ok = p.valid()
    s = (p.params['Fint'][ok] + p.params['Fmv'][ok]).max()
    print(f'{m:<12} max(Fint+Fmv) = {s:.4f}   '
          f'{\"softmax\" if s > 0.6 else \"sigmoid\"}')
"
```

**`use_ordered_diffusion`** leaves no trace in the weights, since it changes
only an activation. It is visible in the outputs through the floor it
implies. Without it each diffusivity has a hard lower bound of its own, so
`Dint >= 0.0015`. With it, `Dint = Dpar + softplus(...)`, so the floor is
inherited from `Dpar` and `Dint` can fall below 0.0015.

```bash
python -c "
from pace import common as C
for m in ['dnn','pace','cnn_fusion']:
    p = C.load_result(m, 25, 19); ok = p.valid()
    d = p.params['Dint'][ok].min()
    print(f'{m:<12} min Dint = {d:.6f}   '
          f'{\"ordered\" if d < 0.0015 else \"independent floors\"}')
"
```

Note that checking `Dpar < Dint < Dmv` does **not** distinguish the two.
The bound ranges are disjoint and contiguous, so that ordering holds under
either parameterisation.

---

## What is not included

The in-vivo cohort is protected health information and cannot be
redistributed. The in-vivo code and the synthetic brain map experiments are
therefore outside this repository.

`synthetic_IVIM_train_2D.h5`, the training set, is also not included. It is
needed only to retrain from scratch, not to reproduce anything described
here, and it is available from the authors on request.

## If something does not match

Please open an issue at
https://github.com/gavtoski/PACE_IVIM/issues, including the command you
ran, the output you saw, and the output of:

```bash
python -c "import sys, torch, numpy; print(sys.version); print(torch.__version__, numpy.__version__)"
```
