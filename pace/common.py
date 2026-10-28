"""Shared helpers for the PACE synthetic evaluation."""

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


@dataclass(frozen=True)
class Paths:
    """Resolved repository layout."""
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
    """Walk upward from `start` looking for configs/models.json."""
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
    """Minimal YAML reader for the flat two level configs/paths.yaml."""
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
    """Build a Paths object.

    The data folder is PACE_DATA_DIR when set, else data: dir: in
    configs/paths.yaml, else data/synthetic in the repository. Every data
    file is read from that folder; paths.yaml may still name a single
    file elsewhere, unless PACE_DATA_DIR is set.
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

    env_data = os.environ.get("PACE_DATA_DIR")
    data_dir = resolve(env_data or data_cfg.get("dir"), "data/synthetic")

    def data_file(key, name):
        value = None if env_data else data_cfg.get(key)
        return resolve(value, data_dir / name)

    return Paths(
        repo_root=root,
        configs_dir=root / "configs",
        data_dir=data_dir,
        checkpoints_dir=resolve(ck_cfg.get("synthetic"), "checkpoints/synthetic"),
        results_dir=resolve(rs_cfg.get("synthetic"), "results/synthetic"),
        figures_dir=resolve(cfg.get("figures"), "figures"),
        synthetic_test=data_file("synthetic_test", "synthetic_IVIM_test_2D.h5"),
        synthetic_val=data_file("synthetic_val", "synthetic_IVIM_val_2D.h5"),
        bvals=data_file("bvals", "bval46.bval"),
    )


@dataclass(frozen=True)
class Checkpoint:
    """One trained network, as recorded in configs/models.json."""
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
    """Model registry read from configs/models.json."""

    def __init__(self, doc: dict, paths: Paths):
        self._paths = paths
        self.schema_version = doc.get("schema_version", 1)
        self.domain = doc.get("domain", "synthetic")

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

    def checkpoint(self, model: str, snr: int, seed: int,
                   allow_seed_fallback: bool = False) -> Checkpoint:
        """The manifest entry for one checkpoint."""
        try:
            return self._index[(model, int(snr), int(seed))]
        except KeyError:
            pass

        if allow_seed_fallback:
            siblings = [c for c in self._checkpoints
                        if c.model == model and c.snr_db == int(snr)]
            if siblings:
                import dataclasses
                template = siblings[0]
                print(f"[MANIFEST] {model} snr{snr} seed{seed} is not in the "
                      f"manifest; taking architecture from "
                      f"{template.public_name}")
                return dataclasses.replace(
                    template, seed=int(seed),
                    public_name=f"{model}_snr{int(snr)}_seed{int(seed)}")

        raise KeyError(
            f"no checkpoint for model={model!r} snr={snr} seed={seed}. "
            f"Available SNRs for {model!r}: {self.snrs(model)}") from None

    def cfg_for(self, model: str) -> tuple[str, str]:
        """Return (cfg_name, fusion_mode) for a neural model."""
        for c in self._checkpoints:
            if c.model == model:
                return c.cfg_name, c.fusion_mode
        raise KeyError(f"{model!r} is not a neural model in this manifest")


    def verify(self, check_hash: bool = False) -> list[str]:
        """Return a list of problems."""
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


ARCH_CONFIGS = {
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


def patch_size_from_state(state, default=3):
    """Anatomical patch size implied by a checkpoint's weights."""
    if not isinstance(state, dict):
        return default
    for key in ("t1_fc.weight", "flair_fc.weight",
                "module.t1_fc.weight", "module.flair_fc.weight"):
        w = state.get(key)
        if w is None:
            continue
        n = int(w.shape[-1]) // 16
        size = int(round(n ** 0.5))
        if size * size == n and size > 0:
            return size
    for key in ("b0_fc.weight", "module.b0_fc.weight"):
        w = state.get(key)
        if w is None:
            continue
        n = int(w.shape[-1]) // 32
        size = int(round(n ** 0.5))
        if size * size == n and size > 0:
            return size
    return default


def widen_bounds_voorter(npars, factor=0.3, verbose=True):
    """Widen the sigmoid bounds, as Voorter et al."""
    lo = [float(v) for v in npars.cons_min]
    hi = [float(v) for v in npars.cons_max]
    span = [factor * (b - a) for a, b in zip(lo, hi)]
    npars.cons_min = [max(a - s, 0.0) for a, s in zip(lo, span)]
    npars.cons_max = [b + s for b, s in zip(hi, span)]
    if verbose:
        names = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]
        pairs = ", ".join(
            f"{n}=[{a:.4f},{b:.4f}]"
            for n, a, b in zip(names, npars.cons_min, npars.cons_max))
        print(f"[BOUNDS] widened by {factor:g} x range, floor clamped at "
              f"zero: {pairs}")
    return npars


def build_net(checkpoint: "Checkpoint", bvals, device=None,
              arch_overrides=None, patch_size=3):
    """Construct an untrained Net matching one checkpoint's architecture."""
    cfg = ARCH_CONFIGS.get(checkpoint.cfg_name)
    if cfg is not None and arch_overrides:
        cfg = dict(cfg)
        cfg.update(arch_overrides)
    if cfg is None:
        raise KeyError(
            f"no ARCH_CONFIGS entry for cfg_name={checkpoint.cfg_name!r} "
            f"(model {checkpoint.model!r}). Known: {sorted(ARCH_CONFIGS)}")

    for flag in ("use_ordered_diffusion", "use_softmax_fractions"):
        if getattr(checkpoint, flag) is None and cfg.get(flag) is None:
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
    _widen = (cfg or {}).get("widen_bounds", 0.0)
    if _widen:
        widen_bounds_voorter(npars, factor=float(_widen), verbose=False)

    return Net(
        bvals=bvals, net_pars=npars, patch_size=int(patch_size),
        spatial_on=cfg["spatial_on"],
        parallel_heads=cfg["parallel_heads"],
        recon_struct=cfg["recon_struct"],
        recon_b0=cfg["recon_b0"],
        use_struct=cfg["use_struct"],
        use_b0=cfg["use_b0"],
        b0_channels=cfg.get("b0_channels", 1),
        ivim_latent_dim=cfg["ivim_latent_dim"],
        latent_dropout_p=cfg["latent_dropout_p"],
        modality_dropout_p=cfg["modality_dropout_p"],
        detach_recon=cfg.get("detach_recon", False),
        detach_spatial_delta=cfg.get("detach_spatial_delta", True),
        use_anat_token=cfg.get("use_anat_token", True),
        gate_inits=cfg.get("gate_inits", None),
        use_ordered_diffusion=(cfg.get("use_ordered_diffusion")
                               if cfg.get("use_ordered_diffusion") is not None
                               else checkpoint.use_ordered_diffusion),
        use_softmax_fractions=(cfg.get("use_softmax_fractions")
                               if cfg.get("use_softmax_fractions") is not None
                               else checkpoint.use_softmax_fractions),
        fusion_mode=(cfg.get("fusion_mode") or checkpoint.fusion_mode),
    ).to(dev)


def _find_checkpoint(root: Path, ck: "Checkpoint") -> Path:
    """Locate one checkpoint under an alternative directory."""
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


SUMMARY_ARCH_KEYS = (
    "spatial_on", "use_struct", "use_b0", "use_anat_token",
    "b0_channels", "widen_bounds", "fusion_mode", "parallel_heads",
    "recon_struct", "recon_b0", "ivim_latent_dim",
    "use_ordered_diffusion", "use_softmax_fractions",
)

_ARCH_INT_KEYS = {"b0_channels", "ivim_latent_dim"}
_ARCH_FLOAT_KEYS = {"widen_bounds"}
_ARCH_STR_KEYS = {"fusion_mode"}


def summary_path_for(ckpt_path):
    """The summary npz written beside one checkpoint, or None."""
    ckpt_path = Path(ckpt_path)
    name = ckpt_path.name
    prefix = "trained_net_"
    if name.startswith(prefix):
        cand = ckpt_path.with_name("summary_" + name[len(prefix):])
        cand = cand.with_suffix(".npz")
        if cand.is_file():
            return cand

    pts = sorted(ckpt_path.parent.glob("*.pt"))
    sums = sorted(ckpt_path.parent.glob("summary_*.npz"))
    if len(pts) == 1 and len(sums) == 1:
        return sums[0]
    return None


def arch_from_summary(ckpt_path):
    """Architecture keys recorded beside a checkpoint, or an empty dict."""
    p = summary_path_for(ckpt_path)
    if p is None:
        return {}
    try:
        doc = np.load(p, allow_pickle=True)
    except Exception as e:
        warnings.warn(f"could not read {p}: {type(e).__name__}: {e}")
        return {}

    out = {}
    for k in SUMMARY_ARCH_KEYS:
        if k not in doc.files:
            continue
        v = doc[k]
        try:
            v = v.item()
        except (AttributeError, ValueError):
            pass
        if v is None:
            continue
        if isinstance(v, float) and v != v:
            continue
        try:
            if k in _ARCH_INT_KEYS:
                v = int(v)
            elif k in _ARCH_FLOAT_KEYS:
                v = float(v)
            elif k in _ARCH_STR_KEYS:
                v = str(v)
            else:
                v = bool(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    return out


def load_net(model: str, snr: int, seed: int, bvals=None, device=None,
             paths: Paths | None = None, strict: bool = True,
             checkpoint_dir=None, arch_overrides=None, patch_size=None):
    """Build a Net and load its trained weights."""
    import torch

    p = paths or load_paths()
    manifest = load_manifest(p)
    ck = manifest.checkpoint(model, snr, seed,
                             allow_seed_fallback=checkpoint_dir is not None)

    if bvals is None:
        bvals = load_bvals(p)

    if checkpoint_dir is not None:
        path = _find_checkpoint(Path(checkpoint_dir), ck)
    else:
        path = ck.path(p)
        if not path.is_file():
            path = _find_checkpoint(p.checkpoints_dir, ck)

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    state = torch.load(path, map_location=resolve_device(device),
                       weights_only=True)

    if patch_size is None:
        patch_size = patch_size_from_state(state)

    recorded = arch_from_summary(path)
    if recorded:
        merged = dict(recorded)
        merged.update(arch_overrides or {})
        arch_overrides = merged
        base = ARCH_CONFIGS.get(ck.cfg_name) or {}
        disagree = {k: v for k, v in recorded.items()
                    if k in base and base[k] != v}
        if disagree:
            said = "; ".join(f"{k}: manifest {base[k]}, summary {v}"
                             for k, v in sorted(disagree.items()))
            print(f"[ARCH] {ck.public_name}: {said}")

    net = build_net(ck, bvals, device, arch_overrides=arch_overrides,
                    patch_size=patch_size)
    try:
        net.load_state_dict(state, strict=strict)
    except RuntimeError as e:
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


MODEL_COLORS = {
    "dnn":                   "#7eb8da",
    "pace":                  "#8338ec",
    "cnn_fusion":            "#e76f51",
    "lsq":                   "#2a9d8f",
    "nnls":                  "#d4a017",
    "map":                   "#8c510a",
    "pace_recon":            "#264653",
    "pace_nosimplex":        "#8d99ae",
    "pace_recon_nosimplex":  "#4a4e69",
    "cnn_fusion_nosimplex":  "#b08968",
    "dnn_simplex":           "#c2185b",
    "ivim3brain":            "#14375e",
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
    "ivim3brain":            "IVIM3C",
}

MODEL_ORDER = [
    "pace", "pace_recon", "pace_nosimplex", "pace_recon_nosimplex",
    "cnn_fusion", "cnn_fusion_nosimplex",
    "dnn", "dnn_simplex",
    "ivim3brain",
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


PARAM_NAMES = ["Dpar", "Dint", "Dmv", "Fint", "Fmv", "S0"]

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
    """Three compartment IVIM forward model."""
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


def signal_rmse_per_voxel(pred, ref) -> np.ndarray:
    """Per voxel RMSE across b-values."""
    n = min(pred.shape[0], ref.shape[0])
    return np.sqrt(np.nanmean((pred[:n] - ref[:n]) ** 2, axis=1))


def param_rmse(pred, gt) -> float:
    """Pooled RMSE over voxels, sqrt of the mean squared error."""
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
    """Flag voxels whose parameters cannot produce a meaningful signal."""
    n = max((len(v) for v in params.values()), default=0)
    bad = np.zeros(n, dtype=bool)
    for name, v in params.items():
        arr = np.asarray(v, dtype=np.float64)
        bad |= ~np.isfinite(arr)
        if name.startswith("D"):
            bad |= arr < 0
    return bad


def boundary_mask(params: dict) -> np.ndarray:
    """Flag voxels where a diffusion coefficient sits at exactly zero."""
    n = max((len(v) for v in params.values()), default=0)
    at_bound = np.zeros(n, dtype=bool)
    for name, v in params.items():
        if name.startswith("D"):
            at_bound |= np.asarray(v, dtype=np.float64) == 0.0
    return at_bound


_RESULT_DIR_RE = re.compile(r"^(?P<model>.+)_snr(?P<snr>\d+)$")
_RESULT_FILE_RE = re.compile(r"^inferred_seed(?P<seed>\d+)\.h5$")


def _match_key(available, param, suffix):
    """Find the H5 key for a parameter, tolerating case differences."""
    want = f"{param}_{suffix}".lower()
    for k in available:
        if k.lower() == want:
            return k
    return None


@dataclass
class Prediction:
    """One inference result: predictions, ground truth and tissue labels."""
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
        """Voxels usable for analysis."""
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
    """Map (model, snr) to {seed: path} for everything under results/."""
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
    """Load the global ground truth arrays from the synthetic dataset."""
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
    """Reference CNR computed once from the global ground truth."""
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
    """Signal level reconstruction accuracy."""
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
    """Per parameter estimation error, split by tissue."""
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
    """WMH against NAWM contrast to noise ratio, predicted and ground truth."""
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
    """Darken a colour toward black, returning an RGB tuple."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (r * factor, g * factor, b * factor)


def darken_hex(hex_color, factor=0.6):
    """Darken a hex colour, returning a hex string."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    dark = tuple(int(c * factor) for c in rgb)
    return f"#{dark[0]:02x}{dark[1]:02x}{dark[2]:02x}"
