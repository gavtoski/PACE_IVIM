"""Shared helpers for the PACE synthetic evaluation.

This module carries everything the figure scripts need and nothing they do
not. It has no dependency on torch or on the network definition, so the
figures can be reproduced from the committed results without a GPU or a
deep learning stack.

Layout:
    1. Paths            repository layout, from configs/paths.yaml
    2. Manifest         model registry, from configs/models.json
    3. Styles           colours and labels, derived from the manifest
    4. Tissue           segmentation decoding
    5. Physics          three compartment IVIM forward model
    6. Metrics          RMSE, bias, CNR, Cohen's d
    7. Result I/O       discovery and loading, schema detecting
    8. Frames           the three analysis DataFrames
    9. Plotting         MRM figure formatting

Metric note. Signal RMSE and parameter RMSE aggregate differently, and
this is deliberate rather than an inconsistency. Signal RMSE has an inner
axis to collapse, the b-values, so it takes the square root per voxel and
then averages over voxels. Parameter RMSE has no inner axis, so it pools
over voxels before taking the square root. Both formulas are carried here
unchanged from the original analysis; they are given distinct names so the
two can never be confused for one another.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "Paths", "load_paths", "find_repo_root",
    "Checkpoint", "Manifest", "load_manifest",
    "ARCH_CONFIGS", "build_net", "load_net", "resolve_device",
    "MODEL_COLORS", "MODEL_LABELS", "MODEL_ORDER", "CONVENTIONAL_MODELS",
    "style_of", "color_of", "label_of", "ordered_models",
    "marker_of", "marker_size_of", "linestyle_of", "clean_spines",
    "PARAM_LATEX", "ground_truth_cnr",
    "TISSUE_COLORS", "TISSUE_ORDER", "decode_tissue", "decode_tissue_array",
    "PARAM_NAMES", "PARAM_ORDER_FIGURE", "PARAM_UNITS",
    "ivim_signal_3c", "reconstruct_signal",
    "signal_rmse_per_voxel", "param_rmse", "param_bias",
    "cnr_magnotta", "cohens_d", "nonphysical_mask", "boundary_mask",
    "Prediction", "discover_results", "load_result", "load_bvals",
    "load_ground_truth", "result_schema_report",
    "build_signal_rmse_df", "build_param_rmse_df", "build_cnr_df",
    "MRM_RC", "MRM_SINGLE_COL_MM", "MRM_DOUBLE_COL_MM",
    "apply_mrm_style", "figure_size", "save_figure",
    "darken", "darken_hex",
]


# ===========================================================
# 1. Paths
# ===========================================================

@dataclass(frozen=True)
class Paths:
    """Resolved repository layout. All fields are absolute."""
    repo_root: Path
    configs_dir: Path
    data_dir: Path
    checkpoints_dir: Path
    results_dir: Path
    figures_dir: Path
    synthetic_test: Path
    synthetic_val: Path
    bvals: Path

    def ground_truth(self, split: str = "test") -> Path:
        if split == "test":
            return self.synthetic_test
        if split == "val":
            return self.synthetic_val
        raise ValueError(f"split must be 'test' or 'val', got {split!r}")


def find_repo_root(start=None) -> Path:
    """Walk upward from `start` looking for configs/models.json.

    Falls back to the PACE_IVIM_ROOT environment variable. This lets the
    scripts run from any working directory.
    """
    env = os.environ.get("PACE_IVIM_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "configs" / "models.json").is_file():
            return p
        raise FileNotFoundError(
            f"PACE_IVIM_ROOT={p} does not contain configs/models.json")

    here = Path(start).resolve() if start else Path(__file__).resolve()
    for cand in [here, *here.parents]:
        if (cand / "configs" / "models.json").is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate the repository root. Run from inside the "
        "repository, or set PACE_IVIM_ROOT.")


def _read_yaml(path: Path) -> dict:
    """Minimal YAML reader for the flat two level configs/paths.yaml.

    Uses PyYAML when available and falls back to a small parser so that
    the figure scripts do not hard require it.
    """
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        pass

    out, section = {}, None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indented = line[0] in " \t"
            key, _, val = line.strip().partition(":")
            key, val = key.strip(), val.strip().strip("'\"")
            if indented and section is not None:
                out.setdefault(section, {})[key] = val
            elif val:
                out[key] = val
                section = None
            else:
                section = key
                out.setdefault(section, {})
    return out


@lru_cache(maxsize=4)
def load_paths(config_path=None, repo_root=None) -> Paths:
    """Build a Paths object, optionally overridden by configs/paths.yaml.

    Any value in the config may be absolute or repository relative.
    Missing values fall back to the standard layout.
    """
    root = Path(repo_root).resolve() if repo_root else find_repo_root()

    cfg = {}
    cfg_file = Path(config_path) if config_path else root / "configs" / "paths.yaml"
    if cfg_file.is_file():
        cfg = _read_yaml(cfg_file)

    def resolve(value, default):
        if not value:
            return (root / default).resolve()
        p = Path(str(value)).expanduser()
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    ck_cfg = cfg.get("checkpoints", {}) if isinstance(cfg.get("checkpoints"), dict) else {}
    rs_cfg = cfg.get("results", {}) if isinstance(cfg.get("results"), dict) else {}

    data_dir = root / "data" / "synthetic"
    return Paths(
        repo_root=root,
        configs_dir=root / "configs",
        data_dir=data_dir,
        checkpoints_dir=resolve(ck_cfg.get("synthetic"), "checkpoints/synthetic"),
        results_dir=resolve(rs_cfg.get("synthetic"), "results/synthetic"),
        figures_dir=resolve(cfg.get("figures"), "figures"),
        synthetic_test=resolve(data_cfg.get("synthetic_test"),
                               "data/synthetic/synthetic_IVIM_test_2D.h5"),
        synthetic_val=resolve(data_cfg.get("synthetic_val"),
                              "data/synthetic/synthetic_IVIM_val_2D.h5"),
        bvals=resolve(data_cfg.get("bvals"), "data/synthetic/bval46.bval"),
    )


# ===========================================================
# 2. Manifest
# ===========================================================

@dataclass(frozen=True)
class Checkpoint:
    """One trained network, as recorded in configs/models.json.

    The three architecture flags below are recorded per checkpoint rather
    than derived at run time. In the original analysis they were set from
    three separate places: a dictionary literal, a spread operator, and a
    patch applied roughly 2600 lines after the declaration. A config that
    missed one silently fell back to a default, and because these flags
    change activations rather than parameter shapes, the resulting network
    still loaded a checkpoint without error and returned wrong numbers.
    Recording them here makes a missing value raise instead.
    """
    public_name: str
    model: str
    legacy_tag: str
    cfg_name: str
    fusion_mode: str
    snr_db: int
    seed: int
    target_rel: str
    use_ordered_diffusion: bool | None = None
    use_softmax_fractions: bool | None = None
    sha256: str = ""
    size_bytes: int = 0

    def path(self, paths: Paths) -> Path:
        return paths.repo_root / self.target_rel


class Manifest:
    """Model registry read from configs/models.json.

    This replaces filename parsing entirely. Nothing downstream should
    infer a model, SNR or seed from a path.
    """

    def __init__(self, doc: dict, paths: Paths):
        self._paths = paths
        self.schema_version = doc.get("schema_version", 1)
        self.domain = doc.get("domain", "synthetic")

        # conventional_models maps legacy tag -> public name.
        self._conventional = sorted(doc.get("conventional_models", {}).values())

        self._checkpoints: list[Checkpoint] = []
        for e in doc.get("checkpoints", []):
            self._checkpoints.append(Checkpoint(
                public_name=e["public_name"],
                model=e["model"],
                legacy_tag=e.get("legacy_tag", ""),
                cfg_name=e.get("cfg_name", ""),
                fusion_mode=e.get("fusion_mode", "attention"),
                snr_db=int(e["snr_db"]),
                seed=int(e["seed"]),
                target_rel=e["target_rel"],
                use_ordered_diffusion=e.get("use_ordered_diffusion"),
                use_softmax_fractions=e.get("use_softmax_fractions"),
                sha256=e.get("sha256") or "",
                size_bytes=int(e.get("size_bytes", 0)),
            ))

        self._index = {(c.model, c.snr_db, c.seed): c for c in self._checkpoints}

    # ---- membership ----

    @property
    def neural_models(self) -> list[str]:
        return sorted({c.model for c in self._checkpoints})

    @property
    def conventional_models(self) -> list[str]:
        return list(self._conventional)

    @property
    def all_models(self) -> list[str]:
        return ordered_models(self.neural_models + self.conventional_models)

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return list(self._checkpoints)

    def is_neural(self, model: str) -> bool:
        return model in self.neural_models

    # ---- coverage ----

    def snrs(self, model: str | None = None) -> list[int]:
        cs = self._checkpoints if model is None else [
            c for c in self._checkpoints if c.model == model]
        return sorted({c.snr_db for c in cs})

    def seeds(self, model: str | None = None, snr: int | None = None) -> list[int]:
        cs = self._checkpoints
        if model is not None:
            cs = [c for c in cs if c.model == model]
        if snr is not None:
            cs = [c for c in cs if c.snr_db == snr]
        return sorted({c.seed for c in cs})

    def checkpoint(self, model: str, snr: int, seed: int) -> Checkpoint:
        try:
            return self._index[(model, int(snr), int(seed))]
        except KeyError:
            raise KeyError(
                f"no checkpoint for model={model!r} snr={snr} seed={seed}. "
                f"Available SNRs for {model!r}: {self.snrs(model)}") from None

    def cfg_for(self, model: str) -> tuple[str, str]:
        """Return (cfg_name, fusion_mode) for a neural model."""
        for c in self._checkpoints:
            if c.model == model:
                return c.cfg_name, c.fusion_mode
        raise KeyError(f"{model!r} is not a neural model in this manifest")

    # ---- integrity ----

    def verify(self, check_hash: bool = False) -> list[str]:
        """Return a list of problems. Empty means everything checks out."""
        import hashlib
        problems = []
        for c in self._checkpoints:
            p = c.path(self._paths)
            if not p.is_file():
                problems.append(f"missing: {c.target_rel}")
                continue
            if c.size_bytes and p.stat().st_size != c.size_bytes:
                problems.append(f"size mismatch: {c.target_rel}")
                continue
            if check_hash and c.sha256:
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        h.update(b)
                if h.hexdigest() != c.sha256:
                    problems.append(f"sha256 mismatch: {c.target_rel}")
        return problems

    def __repr__(self):
        return (f"<Manifest {self.domain}: {len(self._checkpoints)} checkpoints, "
                f"neural={self.neural_models}, "
                f"conventional={self.conventional_models}>")


@lru_cache(maxsize=4)
def load_manifest(paths: Paths | None = None) -> Manifest:
    p = paths or load_paths()
    with open(p.configs_dir / "models.json", encoding="utf-8") as fh:
        return Manifest(json.load(fh), p)


# ===========================================================
# 3. Network construction
# ===========================================================

# Architecture parameters per model, excluding the three flags that are
# recorded per checkpoint in the manifest. Keys are the cfg_name values
# stored in configs/models.json, which are the original analysis tags.
#
# These are the arguments Net.__init__ accepts. Anything absent here takes
# the Net default, matching how the analysis constructed its networks.
ARCH_CONFIGS = {
    # Signal only baseline. No spatial pathway at all, so the anatomical
    # token, structural and b0 inputs, and both dropout terms are off.
    "R13_LongDNN": {
        "spatial_on": False,
        "use_struct": False,
        "use_b0": False,
        "parallel_heads": True,
        "recon_struct": False,
        "recon_b0": False,
        "ivim_latent_dim": 32,
        "latent_dropout_p": 0.0,
        "modality_dropout_p": 0.0,
        "detach_recon": False,
        "detach_spatial_delta": True,
        "use_anat_token": False,
    },
    # Cross attention fusion.
    #
    # recon_struct and recon_b0 are True here even though the analysis
    # trained with alpha_recon=0, meaning the reconstruction loss was never
    # applied. The training driver sets recon_all=False but leaves the two
    # individual flags alone, and learn_IVIM defaults both to True, so the
    # decoders were built and saved into every checkpoint. The inference
    # configuration in the original notebook set them False and relied on
    # load_state_dict(strict=False) to discard the extra weights. That was
    # harmless, because the decoders are terminal: they consume the spatial
    # token after the parameter heads have already produced their output and
    # never feed back. Matching the trained architecture instead lets the
    # checkpoint load strictly, so a genuine mismatch would be caught.
    "R29_CE_ID": {
        "spatial_on": True,
        "use_struct": True,
        "use_b0": True,
        "parallel_heads": True,
        "recon_struct": True,
        "recon_b0": True,
        "ivim_latent_dim": 32,
        "latent_dropout_p": 0.3,
        "modality_dropout_p": 0.15,
        "detach_recon": False,
        "detach_spatial_delta": True,
        "use_anat_token": True,
    },
    # Concatenation fusion. Identical to R29_CE_ID apart from fusion_mode,
    # which is recorded in the manifest, so the pair isolates the fusion
    # module as the single architectural variable.
    "R36_ConcatSoftmax": {
        "spatial_on": True,
        "use_struct": True,
        "use_b0": True,
        "parallel_heads": True,
        "recon_struct": True,
        "recon_b0": True,
        "ivim_latent_dim": 32,
        "latent_dropout_p": 0.3,
        "modality_dropout_p": 0.15,
        "detach_recon": False,
        "detach_spatial_delta": True,
        "use_anat_token": True,
    },
}


def resolve_device(device=None):
    """Pick a compute device, preferring the accelerator that is present."""
    import torch
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_net(checkpoint: "Checkpoint", bvals, device=None):
    """Construct an untrained Net matching one checkpoint's architecture.

    Mirrors the Net(...) call inside learn_ivim.learn_IVIM exactly, so a
    network built here is structurally identical to the one that produced
    the weights. The three flags come from the manifest and are required;
    a missing value raises rather than falling back to a default.
    """
    cfg = ARCH_CONFIGS.get(checkpoint.cfg_name)
    if cfg is None:
        raise KeyError(
            f"no ARCH_CONFIGS entry for cfg_name={checkpoint.cfg_name!r} "
            f"(model {checkpoint.model!r}). Known: {sorted(ARCH_CONFIGS)}")

    for flag in ("use_ordered_diffusion", "use_softmax_fractions"):
        if getattr(checkpoint, flag) is None:
            raise ValueError(
                f"{checkpoint.public_name} has no {flag} in configs/models.json. "
                f"These flags change the output activation without changing "
                f"any parameter shape, so guessing one would load the weights "
                f"successfully and return wrong values. Add them to the "
                f"manifest.")

    from pace.deep_models import Net
    from pace.hyperparams import net_pars

    dev = resolve_device(device)

    npars = net_pars()
    npars.use_three_compartment = True
    npars.fitS0 = True
    npars.device = dev

    return Net(
        bvals=bvals, net_pars=npars, patch_size=3,
        spatial_on=cfg["spatial_on"],
        parallel_heads=cfg["parallel_heads"],
        recon_struct=cfg["recon_struct"],
        recon_b0=cfg["recon_b0"],
        use_struct=cfg["use_struct"],
        use_b0=cfg["use_b0"],
        ivim_latent_dim=cfg["ivim_latent_dim"],
        latent_dropout_p=cfg["latent_dropout_p"],
        modality_dropout_p=cfg["modality_dropout_p"],
        detach_recon=cfg.get("detach_recon", False),
        detach_spatial_delta=cfg.get("detach_spatial_delta", True),
        use_anat_token=cfg.get("use_anat_token", True),
        gate_inits=cfg.get("gate_inits", None),
        use_ordered_diffusion=checkpoint.use_ordered_diffusion,
        use_softmax_fractions=checkpoint.use_softmax_fractions,
        fusion_mode=checkpoint.fusion_mode,
    ).to(dev)


def _find_checkpoint(root: Path, ck: "Checkpoint") -> Path:
    """Locate one checkpoint under an alternative directory.

    The release stores weights flat, one .pt per public name. Training
    writes a directory per run containing a .pt whose filename carries the
    legacy tag. Both layouts are searched so either can be evaluated.
    """
    flat = root / f"{ck.public_name}.pt"
    if flat.is_file():
        return flat

    run_dir = root / ck.public_name
    if run_dir.is_dir():
        hits = sorted(run_dir.glob("*.pt"))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(
                f"{run_dir} contains {len(hits)} .pt files, so which one to "
                f"load is ambiguous: {[h.name for h in hits]}")

    raise FileNotFoundError(
        f"no checkpoint for {ck.public_name} under {root}. Looked for "
        f"{flat.name} and {ck.public_name}/*.pt")


def load_net(model: str, snr: int, seed: int, bvals=None, device=None,
             paths: Paths | None = None, strict: bool = True,
             checkpoint_dir=None):
    """Build a Net and load its trained weights. Returns (net, checkpoint).

    Strict by default. The original inference code used strict=False, which
    means a network built without the softmax simplex will happily load a
    checkpoint trained with it and silently discard Fpar_head, giving
    independent sigmoid fractions instead of a simplex. That produces
    plausible numbers and no error. Strict loading turns it into a crash.

    checkpoint_dir overrides where the weights are read from, while the
    architecture still comes from the manifest. This is what lets a
    retrained network be evaluated against the released one: the two are
    built identically and differ only in their weights. Two layouts are
    accepted, the flat one used by the release and the per run
    subdirectory that training writes:

        <dir>/<model>_snr<N>_seed<S>.pt
        <dir>/<model>_snr<N>_seed<S>/*.pt
    """
    import torch

    p = paths or load_paths()
    manifest = load_manifest(p)
    ck = manifest.checkpoint(model, snr, seed)

    if bvals is None:
        bvals = load_bvals(p)

    if checkpoint_dir is None:
        path = ck.path(p)
    else:
        path = _find_checkpoint(Path(checkpoint_dir), ck)

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    net = build_net(ck, bvals, device)
    state = torch.load(path, map_location=resolve_device(device),
                       weights_only=True)
    try:
        net.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        # Summarise by layer prefix rather than listing every tensor, so the
        # message names the module that differs instead of scrolling.
        have = set(state)
        want = set(net.state_dict())

        def prefixes(keys):
            out = {}
            for k in keys:
                out[k.split(".")[0]] = out.get(k.split(".")[0], 0) + 1
            return ", ".join(f"{k} ({n})" for k, n in sorted(out.items()))

        extra, missing = have - want, want - have
        detail = []
        if extra:
            detail.append(f"  in the checkpoint but not the built net: "
                          f"{prefixes(extra)}")
        if missing:
            detail.append(f"  in the built net but not the checkpoint: "
                          f"{prefixes(missing)}")
        raise RuntimeError(
            f"{ck.public_name} weights do not match the architecture built "
            f"from cfg_name={ck.cfg_name!r}, fusion_mode={ck.fusion_mode!r}, "
            f"ordered={ck.use_ordered_diffusion}, "
            f"softmax={ck.use_softmax_fractions}.\n"
            + "\n".join(detail) + f"\n{e}") from None

    net.eval()
    return net, ck


# ===========================================================
# 4. Styles
# ===========================================================

# Colours preserved from the original analysis so published figures keep
# their identity across the rename. Entries beyond the headline six are
# present so that adding ablation checkpoints needs no code change.
MODEL_COLORS = {
    "dnn":                   "#7eb8da",
    "pace":                  "#8338ec",
    "cnn_fusion":            "#e76f51",
    "lsq":                   "#2a9d8f",
    "nnls":                  "#d4a017",
    # MAP is brown rather than the #6a4c93 purple used in older versions
    # of the analysis, which separates it from PACE.
    "map":                   "#8c510a",
    # ablation variants
    "pace_recon":            "#264653",
    "pace_nosimplex":        "#8d99ae",
    "pace_recon_nosimplex":  "#4a4e69",
    "cnn_fusion_nosimplex":  "#b08968",
    "dnn_simplex":           "#c2185b",
}

MODEL_LABELS = {
    "dnn":                   "DNN",
    "pace":                  "PACE",
    "cnn_fusion":            "CNN Fusion",
    "lsq":                   "LSQ",
    "nnls":                  "NNLS",
    "map":                   "MAP",
    "pace_recon":            "PACE (Full)",
    "pace_nosimplex":        "PACE (No SM)",
    "pace_recon_nosimplex":  "PACE (No SM Full)",
    "cnn_fusion_nosimplex":  "CNN Fusion (No SM)",
    "dnn_simplex":           "DNN (SM)",
}

# Plot order, matching FIG1_MODEL_ORDER, FIG2_MODEL_ORDER and
# FIG3_MODEL_ORDER in the analysis: the proposed models first, then the
# baselines. Ablation variants sit beside the family they belong to.
MODEL_ORDER = [
    "pace", "pace_recon", "pace_nosimplex", "pace_recon_nosimplex",
    "cnn_fusion", "cnn_fusion_nosimplex",
    "dnn", "dnn_simplex",
    "lsq", "nnls", "map",
]

CONVENTIONAL_MODELS = {"lsq", "nnls", "map"}

_HEADLINE = {"pace", "cnn_fusion"}

_FALLBACK_COLOR = "#888888"


def style_of(model: str) -> dict:
    """Matplotlib keyword arguments for one model."""
    if model in CONVENTIONAL_MODELS:
        ls, lw = ":", 1.8
    elif model in _HEADLINE:
        ls, lw = "-", 2.5
    else:
        ls, lw = "-", 2.0
    return {
        "color": MODEL_COLORS.get(model, _FALLBACK_COLOR),
        "linestyle": ls,
        "linewidth": lw,
        "label": MODEL_LABELS.get(model, model),
    }


def marker_of(model: str) -> str:
    """Triangles for conventional fitters, circles for learned models."""
    return "^" if model in CONVENTIONAL_MODELS else "o"


def marker_size_of(model: str) -> int:
    """Triangles read smaller than circles at equal point size."""
    return 6 if model in CONVENTIONAL_MODELS else 5


def linestyle_of(model: str) -> str:
    return "--" if model in CONVENTIONAL_MODELS else "-"


def clean_spines(ax, top=False, right=False):
    """Hide the top and right spines and add a subtle grid."""
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.grid(True, alpha=0.15)


def color_of(model: str) -> str:
    return MODEL_COLORS.get(model, _FALLBACK_COLOR)


def label_of(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def ordered_models(models) -> list[str]:
    """Sort models into the canonical plot order, unknowns last."""
    rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(set(models), key=lambda m: (rank.get(m, len(rank)), m))


# ===========================================================
# 5. Tissue
# ===========================================================

TISSUE_COLORS = {
    "GM":   "#1f77b4",
    "NAWM": "#2ca02c",
    "WMH":  "#d62728",
}

TISSUE_ORDER = ["GM", "NAWM", "WMH"]


def decode_tissue(tval, lval) -> str:
    """Scalar tissue decoding."""
    tval, lval = int(tval), int(lval)
    if tval == 2:
        return "GM"
    if tval == 3 and lval == 0:
        return "NAWM"
    if lval != 0:
        return "WMH"
    return "Other"


def decode_tissue_array(tissue_arr, lesion_arr) -> np.ndarray:
    """Vectorised tissue decoding to an object array of labels."""
    t = np.asarray(tissue_arr, dtype=int)
    l = np.asarray(lesion_arr, dtype=int)
    out = np.full(t.shape, "Other", dtype=object)
    out[t == 2] = "GM"
    out[(t == 3) & (l == 0)] = "NAWM"
    out[l != 0] = "WMH"
    return out


# ===========================================================
# 6. Physics
# ===========================================================

PARAM_NAMES = ["Dpar", "Dint", "Dmv", "Fint", "Fmv", "S0"]

# Figure column order groups by compartment: parenchymal, interstitial,
# then microvascular.
PARAM_ORDER_FIGURE = ["Dpar", "Dint", "Fint", "Dmv", "Fmv", "S0"]

PARAM_LATEX = {
    "Dpar": r"$D_\mathrm{par}$",
    "Dint": r"$D_\mathrm{int}$",
    "Dmv":  r"$D_\mathrm{mv}$",
    "Fint": r"$f_\mathrm{int}$",
    "Fmv":  r"$f_\mathrm{mv}$",
    "S0":   r"$S_0$",
}

PARAM_UNITS = {
    "Dpar": "mm$^2$/s", "Dint": "mm$^2$/s", "Dmv": "mm$^2$/s",
    "Fint": "", "Fmv": "", "S0": "a.u.",
}


def ivim_signal_3c(S0, Dpar, Dint, Dmv, Fint, Fmv, bvals) -> np.ndarray:
    """Three compartment IVIM forward model.

    All parameter arrays are shape (N,), bvals is (nb,). Returns (N, nb).
    The parenchymal fraction is the simplex remainder, clipped to [0, 1].
    """
    S0 = np.asarray(S0, dtype=np.float64)
    Dpar = np.asarray(Dpar, dtype=np.float64)
    Dint = np.asarray(Dint, dtype=np.float64)
    Dmv = np.asarray(Dmv, dtype=np.float64)
    Fint = np.asarray(Fint, dtype=np.float64)
    Fmv = np.asarray(Fmv, dtype=np.float64)
    b = np.asarray(bvals, dtype=np.float64)

    fp = np.clip(1.0 - Fint - Fmv, 0.0, 1.0)
    S = S0[:, None] * (
        fp[:, None]   * np.exp(-b[None] * Dpar[:, None]) +
        Fint[:, None] * np.exp(-b[None] * Dint[:, None]) +
        Fmv[:, None]  * np.exp(-b[None] * Dmv[:, None])
    )
    return S.astype(np.float32)


def reconstruct_signal(params: dict, bvals) -> np.ndarray:
    """Forward model from a dict keyed by the canonical parameter names."""
    missing = [p for p in PARAM_NAMES if p not in params]
    if missing:
        raise KeyError(f"reconstruct_signal is missing parameters: {missing}")
    return ivim_signal_3c(
        params["S0"], params["Dpar"], params["Dint"],
        params["Dmv"], params["Fint"], params["Fmv"], bvals)


# ===========================================================
# 7. Metrics
# ===========================================================

def signal_rmse_per_voxel(pred, ref) -> np.ndarray:
    """Per voxel RMSE across b-values. Shape (N, nb) to (N,).

    Carried unchanged from the original analysis. Callers average the
    result over voxels, giving mean of per voxel RMSE.
    """
    n = min(pred.shape[0], ref.shape[0])
    return np.sqrt(np.nanmean((pred[:n] - ref[:n]) ** 2, axis=1))


def param_rmse(pred, gt) -> float:
    """Pooled RMSE over voxels, sqrt of the mean squared error.

    Carried unchanged from the original analysis. Note that this pools
    before taking the root, unlike signal_rmse_per_voxel.
    """
    err = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2)))


def param_bias(pred, gt) -> float:
    """Mean signed error, predicted minus ground truth."""
    err = np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan")
    return float(np.mean(err))


def cnr_magnotta(group1, group2) -> float:
    """CNR = |mu1 - mu2| / sqrt(sigma1^2 + sigma2^2), Magnotta 2006, ddof=1."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return float("nan")
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    denom = np.sqrt(s1 ** 2 + s2 ** 2)
    if denom < 1e-30:
        return np.inf if abs(m1 - m2) > 1e-15 else 0.0
    return float(abs(m1 - m2) / denom)


def cohens_d(group1, group2) -> float:
    """Cohen's d with pooled standard deviation, ddof=1."""
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled < 1e-30:
        return float("nan")
    return float((np.mean(g1) - np.mean(g2)) / pooled)


def nonphysical_mask(params: dict) -> np.ndarray:
    """Flag voxels whose parameters cannot produce a meaningful signal.

    A strictly negative diffusion coefficient turns exp(-b*D) into a
    growing exponential. At b = 1000 a value of D = -0.04 yields roughly
    1e17, which is large but finite, so it survives an isfinite filter and
    would dominate a mean taken over thousands of voxels.

    Note that D exactly zero is NOT flagged here. It gives exp(0) = 1, a
    bounded contribution, and simply means that compartment collapsed. It
    is a constrained fit resting on its lower bound, not a hazard. Use
    boundary_mask to count those separately.

    This function only reports. It never alters values, so metrics
    computed with and without consulting it are identical.
    """
    n = max((len(v) for v in params.values()), default=0)
    bad = np.zeros(n, dtype=bool)
    for name, v in params.items():
        arr = np.asarray(v, dtype=np.float64)
        bad |= ~np.isfinite(arr)
        if name.startswith("D"):
            bad |= arr < 0
    return bad


def boundary_mask(params: dict) -> np.ndarray:
    """Flag voxels where a diffusion coefficient sits at exactly zero.

    Typical of a bounded least squares fit resting on its lower bound,
    meaning that compartment contributed nothing. Numerically harmless,
    but a useful indicator of fit degeneracy, and more frequent at low
    SNR. Reported without warning.
    """
    n = max((len(v) for v in params.values()), default=0)
    at_bound = np.zeros(n, dtype=bool)
    for name, v in params.items():
        if name.startswith("D"):
            at_bound |= np.asarray(v, dtype=np.float64) == 0.0
    return at_bound


# ===========================================================
# 8. Result I/O
# ===========================================================

_RESULT_DIR_RE = re.compile(r"^(?P<model>.+)_snr(?P<snr>\d+)$")
_RESULT_FILE_RE = re.compile(r"^inferred_seed(?P<seed>\d+)\.h5$")


def _match_key(available, param, suffix):
    """Find the H5 key for a parameter, tolerating case differences.

    The writers are inconsistent: predictions use Fint_pred and Fmv_pred
    while the ground truth uses fint_gt and fmv_gt. Matching case
    insensitively removes the need to hardcode either convention.
    """
    want = f"{param}_{suffix}".lower()
    for k in available:
        if k.lower() == want:
            return k
    return None


@dataclass
class Prediction:
    """One inference result: predictions, ground truth and tissue labels.

    Two writers produced the committed results and their schemas differ.
    The neural writer stores Valid_mask and bvals but no ground truth; the
    conventional writer embeds ground truth and Signal_noisy but neither
    of the other two. Both ran on the same test split, so ground truth for
    the neural files is taken from the global arrays, aligned row for row.

    `gt_source` records which of the two applied.
    """
    model: str
    snr: int
    seed: int
    params: dict = field(default_factory=dict)
    gt: dict = field(default_factory=dict)
    tissue: np.ndarray | None = None
    lesion: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    bvals: np.ndarray | None = None
    gt_source: str = "none"
    path: Path | None = None

    @property
    def n_voxels(self) -> int:
        for v in self.params.values():
            return int(len(v))
        return 0

    @property
    def has_gt(self) -> bool:
        return len(self.gt) == len(PARAM_NAMES)

    def labels(self) -> np.ndarray:
        """Tissue label strings per voxel."""
        if self.tissue is None or self.lesion is None:
            raise ValueError(f"{self.model} snr{self.snr} seed{self.seed} "
                             f"has no tissue or lesion arrays")
        return decode_tissue_array(self.tissue, self.lesion)

    def mask(self, tissue: str) -> np.ndarray:
        return self.labels() == tissue

    def valid(self) -> np.ndarray:
        """Voxels usable for analysis.

        Combines the writer's Valid_mask, when present, with a finite
        check across every predicted parameter. Voxels rejected during
        inference are normally already NaN, so the two usually agree.
        """
        n = self.n_voxels
        ok = np.ones(n, dtype=bool)
        if self.valid_mask is not None and len(self.valid_mask) == n:
            ok &= self.valid_mask.astype(bool)
        for v in self.params.values():
            ok &= np.isfinite(v)
        return ok

    def _bvals(self, bvals=None):
        if bvals is not None:
            return np.asarray(bvals, dtype=np.float64)
        if self.bvals is not None:
            return np.asarray(self.bvals, dtype=np.float64)
        raise ValueError(
            f"{self.model} snr{self.snr} seed{self.seed} carries no bvals; "
            f"pass them explicitly via load_bvals()")

    def signal(self, bvals=None) -> np.ndarray:
        return reconstruct_signal(self.params, self._bvals(bvals))

    def gt_signal(self, bvals=None) -> np.ndarray:
        if not self.has_gt:
            raise ValueError("ground truth parameters are incomplete")
        return reconstruct_signal(self.gt, self._bvals(bvals))


def discover_results(paths: Paths | None = None) -> dict:
    """Map (model, snr) to {seed: path} for everything under results/.

    Directory names are read only to recover the model and SNR that the
    staging step encoded. Nothing about the architecture is inferred here.
    """
    p = paths or load_paths()
    out: dict = {}
    if not p.results_dir.is_dir():
        return out
    for d in sorted(p.results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _RESULT_DIR_RE.match(d.name)
        if not m:
            continue
        key = (m.group("model"), int(m.group("snr")))
        for f in sorted(d.iterdir()):
            fm = _RESULT_FILE_RE.match(f.name)
            if fm:
                out.setdefault(key, {})[int(fm.group("seed"))] = f
    return out


def load_result(model: str, snr: int, seed: int,
                paths: Paths | None = None,
                gt_split: str = "test") -> Prediction:
    """Load one result H5, detecting which keys it actually carries."""
    import h5py

    p = paths or load_paths()
    path = p.results_dir / f"{model}_snr{int(snr)}" / f"inferred_seed{int(seed)}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"no result file: {path}")

    params, gt = {}, {}
    tissue = lesion = valid = fbvals = None
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        for name in PARAM_NAMES:
            k = _match_key(keys, name, "pred")
            if k is not None:
                params[name] = f[k][:].astype(np.float32)
            k = _match_key(keys, name, "gt")
            if k is not None:
                gt[name] = f[k][:].astype(np.float32)
        for k in keys:
            kl = k.lower()
            if kl == "tissue":
                tissue = f[k][:]
            elif kl == "lesion":
                lesion = f[k][:]
            elif kl == "valid_mask":
                valid = f[k][:]
            elif kl == "bvals":
                fbvals = f[k][:].astype(np.float32)

    missing = [n for n in PARAM_NAMES if n not in params]
    if missing:
        raise KeyError(f"{path.name} is missing predictions for {missing}")

    lengths = {len(v) for v in params.values()}
    if len(lengths) > 1:
        raise ValueError(f"{path.name} has inconsistent prediction lengths: "
                         f"{sorted(lengths)}")
    n = lengths.pop()

    # The neural writer stores no ground truth. Both writers ran on the
    # same split, so the global arrays align row for row.
    gt_source = "embedded"
    if len(gt) < len(PARAM_NAMES):
        gt_source = f"global:{gt_split}"
        g = load_ground_truth(gt_split, p)
        for name in PARAM_NAMES:
            if name in gt:
                continue
            arr = g.get(name)
            if arr is None:
                raise KeyError(
                    f"{path.name} has no {name} ground truth, and none is "
                    f"present in the {gt_split} split either")
            if len(arr) != n:
                raise ValueError(
                    f"{path.name} has {n} voxels but the {gt_split} split "
                    f"has {len(arr)} for {name}. These cannot be aligned.")
            gt[name] = arr
        if tissue is None and "Tissue" in g:
            tissue = g["Tissue"]
        if lesion is None and "Lesion" in g:
            lesion = g["Lesion"]

    if fbvals is None:
        fbvals = load_bvals(p)

    return Prediction(model=model, snr=int(snr), seed=int(seed),
                      params=params, gt=gt, tissue=tissue, lesion=lesion,
                      valid_mask=valid, bvals=fbvals, gt_source=gt_source,
                      path=path)


def load_bvals(paths: Paths | None = None) -> np.ndarray:
    p = paths or load_paths()
    return np.loadtxt(p.bvals).astype(np.float32).ravel()


@lru_cache(maxsize=2)
def load_ground_truth(split: str = "test", paths: Paths | None = None) -> dict:
    """Load the global ground truth arrays from the synthetic dataset.

    Most analysis should prefer the per file ground truth carried inside
    each result, since voxel rejection during inference means the global
    arrays are not aligned to the predictions.
    """
    import h5py

    p = paths or load_paths()
    path = p.ground_truth(split)
    out = {}
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        for name in PARAM_NAMES:
            k = _match_key(keys, name, "gt")
            if k is not None:
                out[name] = f[k][:].astype(np.float32)
        for k in keys:
            if k.lower() in ("tissue", "lesion"):
                out[k.capitalize()] = f[k][:]
    return out


@lru_cache(maxsize=4)
def ground_truth_cnr(pair=("WMH", "NAWM"), split: str = "test",
                     paths: Paths | None = None) -> dict:
    """Reference CNR computed once from the global ground truth.

    The analysis compares every model against a single reference value per
    parameter rather than against a per file recomputation, so this is
    deliberately independent of any model's valid mask.
    """
    g = load_ground_truth(split, paths)
    if "Tissue" not in g or "Lesion" not in g:
        raise KeyError(f"the {split} split carries no Tissue or Lesion arrays")
    labels = decode_tissue_array(g["Tissue"], g["Lesion"])
    a, b = pair
    ma, mb = labels == a, labels == b
    return {p: cnr_magnotta(g[p][ma], g[p][mb])
            for p in PARAM_NAMES if p in g}


def result_schema_report(paths: Paths | None = None) -> dict:
    """Group every result file by its exact key set, for diagnostics."""
    import h5py

    p = paths or load_paths()
    groups: dict = {}
    for (model, snr), seeds in discover_results(p).items():
        for seed, path in seeds.items():
            with h5py.File(path, "r") as f:
                ks = frozenset(f.keys())
            groups.setdefault(ks, []).append(f"{model}_snr{snr}/seed{seed}")
    return groups


# ===========================================================
# 9. Frames
# ===========================================================

def _iter_results(models=None, snrs=None, paths=None):
    found = discover_results(paths)
    for (model, snr), seeds in sorted(found.items()):
        if models is not None and model not in models:
            continue
        if snrs is not None and snr not in snrs:
            continue
        for seed in sorted(seeds):
            yield model, snr, seed


def build_signal_rmse_df(models=None, snrs=None, paths=None,
                         warn_nonphysical=True, gt_split="test"):
    """Signal level reconstruction accuracy.

    Columns: model, snr, seed, rmse, n_voxels, n_nonphysical,
    n_at_bound, gt_source.

    Both the predicted and reference signals are synthesised from their
    respective parameters through the forward model. RMSE is taken per
    voxel across b-values and then averaged over voxels, matching the
    original analysis.
    """
    import pandas as pd

    p = paths or load_paths()
    rows = []
    for model, snr, seed in _iter_results(models, snrs, p):
        pred = load_result(model, snr, seed, p, gt_split=gt_split)
        sig = pred.signal()
        ref = pred.gt_signal()
        vm = (pred.valid()
              & np.isfinite(sig).all(axis=1)
              & np.isfinite(ref).all(axis=1))
        if vm.sum() == 0:
            continue
        n_bad = int(nonphysical_mask(pred.params)[vm].sum())
        n_bound = int(boundary_mask(pred.params)[vm].sum())
        if n_bad and warn_nonphysical:
            warnings.warn(
                f"{model} snr{snr} seed{seed}: {n_bad} of {int(vm.sum())} "
                f"voxels have a strictly negative diffusion coefficient. "
                f"Their reconstructed signal grows exponentially and will "
                f"dominate the mean. The reported value is unaltered; "
                f"inspect the n_nonphysical column.",
                RuntimeWarning, stacklevel=2)
        rows.append({
            "model": model, "snr": snr, "seed": seed,
            "rmse": float(np.mean(signal_rmse_per_voxel(sig[vm], ref[vm]))),
            "n_voxels": int(vm.sum()),
            "n_nonphysical": n_bad,
            "n_at_bound": n_bound,
            "gt_source": pred.gt_source,
        })
    return pd.DataFrame(rows)


def build_param_rmse_df(models=None, snrs=None, tissues=None,
                        min_voxels=10, paths=None, gt_split="test"):
    """Per parameter estimation error, split by tissue.

    Columns: model, snr, seed, param, tissue, rmse, bias, n_voxels.
    RMSE is pooled over voxels, matching the original analysis.
    """
    import pandas as pd

    p = paths or load_paths()
    tissues = list(tissues) if tissues else list(TISSUE_ORDER)
    rows = []
    for model, snr, seed in _iter_results(models, snrs, p):
        pred = load_result(model, snr, seed, p, gt_split=gt_split)
        if pred.tissue is None:
            continue
        labels = pred.labels()
        ok = pred.valid()
        for tissue in tissues:
            tm = (labels == tissue) & ok
            if tm.sum() < min_voxels:
                continue
            for name in PARAM_NAMES:
                a, b = pred.params[name][tm], pred.gt[name][tm]
                rows.append({
                    "model": model, "snr": snr, "seed": seed,
                    "param": name, "tissue": tissue,
                    "rmse": param_rmse(a, b),
                    "bias": param_bias(a, b),
                    "n_voxels": int(tm.sum()),
                })
    return pd.DataFrame(rows)


def build_cnr_df(models=None, snrs=None, pair=("WMH", "NAWM"),
                 min_voxels=10, paths=None, gt_split="test"):
    """WMH against NAWM contrast to noise ratio, predicted and ground truth.

    Columns: model, snr, seed, param, pair, cnr_pred, cnr_gt, cnr_error.
    """
    import pandas as pd

    p = paths or load_paths()
    a_name, b_name = pair
    rows = []
    for model, snr, seed in _iter_results(models, snrs, p):
        pred = load_result(model, snr, seed, p, gt_split=gt_split)
        if pred.tissue is None:
            continue
        labels = pred.labels()
        ok = pred.valid()
        ma, mb = (labels == a_name) & ok, (labels == b_name) & ok
        if ma.sum() < min_voxels or mb.sum() < min_voxels:
            continue
        for name in PARAM_NAMES:
            c_pred = cnr_magnotta(pred.params[name][ma], pred.params[name][mb])
            c_gt = (cnr_magnotta(pred.gt[name][ma], pred.gt[name][mb])
                    if pred.has_gt else float("nan"))
            rows.append({
                "model": model, "snr": snr, "seed": seed,
                "param": name, "pair": f"{a_name}-{b_name}",
                "cnr_pred": c_pred, "cnr_gt": c_gt,
                "cnr_error": c_pred - c_gt,
            })
    return pd.DataFrame(rows)


# ===========================================================
# 10. Plotting
# ===========================================================

MRM_SINGLE_COL_MM = 84
MRM_DOUBLE_COL_MM = 174
_MM_TO_INCH = 1 / 25.4

MRM_RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "savefig.dpi":       600,
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth":   1.0,
}


def apply_mrm_style():
    """Apply Magnetic Resonance in Medicine figure conventions."""
    import matplotlib as mpl
    mpl.rcParams.update(MRM_RC)


def figure_size(width_mm=MRM_DOUBLE_COL_MM, height_mm=None, aspect=0.618):
    """Figure size in inches from a millimetre width."""
    w = width_mm * _MM_TO_INCH
    h = (height_mm * _MM_TO_INCH) if height_mm else w * aspect
    return (w, h)


def save_figure(fig, name, paths=None, formats=("png", "pdf"), dpi=600):
    """Write a figure into figures/ and return the paths written."""
    p = paths or load_paths()
    p.figures_dir.mkdir(parents=True, exist_ok=True)
    stem = name[:-4] if name.lower().endswith((".png", ".pdf", ".svg")) else name
    written = []
    for ext in formats:
        out = p.figures_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        written.append(out)
    return written


def darken(color, factor=0.6):
    """Darken a colour toward black, returning an RGB tuple.

    factor 1.0 leaves it unchanged, 0.0 gives black.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def darken_hex(hex_color, factor=0.6):
    """Darken a hex colour, returning a hex string.

    Carried unchanged from the analysis, including its integer
    truncation, so that annotation and limit-of-agreement line colours
    match the published figures exactly.
    """
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    dark = tuple(int(c * factor) for c in rgb)
    return f"#{dark[0]:02x}{dark[1]:02x}{dark[2]:02x}"
