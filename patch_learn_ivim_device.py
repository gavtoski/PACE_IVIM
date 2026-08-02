#!/usr/bin/env python3
"""Add Apple Silicon GPU support to pace/learn_ivim.py.

The training engine selects its device with

    device = "cuda" if torch.cuda.is_available() else "cpu"

which was written for Colab, where the only options are CUDA and CPU. On
an Apple Silicon machine that falls through to CPU and the GPU sits idle.
This patch adds Metal to the selection and lets the caller override it,
leaving the CUDA and CPU behaviour exactly as it was.

The edit is anchored on the exact source text rather than a line number,
so it is safe to run against a file that has been edited elsewhere. It
refuses to run twice.

DRY RUN BY DEFAULT. A timestamped backup is written before any change.

Usage:
    python patch_learn_ivim_device.py
    python patch_learn_ivim_device.py --execute
    python patch_learn_ivim_device.py --revert
"""

import argparse
import ast
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_REPO = os.path.expanduser("~/Documents/Projects/PACE_IVIM")
TARGET = "pace/learn_ivim.py"

# The signature line to anchor the new parameter against.
SIG_ANCHOR = """    # --- Legacy compat ---
    freeze_perf_gate=False,
):"""

SIG_PATCHED = """    # --- Legacy compat ---
    freeze_perf_gate=False,
    # --- Device selection ---
    device=None,
):"""

BODY_ANCHOR = '''    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True'''

BODY_PATCHED = '''    # Device selection. Passing device explicitly overrides the search.
    # The order below prefers a discrete GPU, then Apple Silicon's Metal
    # backend, then the CPU. Metal was not an option when this function
    # was written, so an unpatched copy falls through to CPU on a Mac.
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and \\
                torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = str(device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True'''


def find_backups(path):
    return sorted(path.parent.glob(path.name + ".bak.*"))


def main():
    ap = argparse.ArgumentParser(
        description="Add Apple Silicon GPU support to the training engine.")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--execute", action="store_true",
                    help="Apply the patch. Without this nothing is written.")
    ap.add_argument("--revert", action="store_true",
                    help="Restore the most recent backup.")
    args = ap.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    path = repo / TARGET

    print("=" * 70)
    print(" Add Metal support to the training engine")
    print("=" * 70)
    print(f" file: {path}")

    if not path.is_file():
        print(f"[ERROR] not found: {path}", file=sys.stderr)
        return 2

    if args.revert:
        backups = find_backups(path)
        if not backups:
            print("[ERROR] no backup to revert to", file=sys.stderr)
            return 2
        shutil.copy2(backups[-1], path)
        print(f" restored from {backups[-1].name}")
        return 0

    src = path.read_text(encoding="utf-8")

    if "device=None" in src and 'elif getattr(torch.backends, "mps"' in src:
        print(" already patched, nothing to do")
        return 0

    problems = []
    if SIG_ANCHOR not in src:
        problems.append("signature anchor not found")
    if BODY_ANCHOR not in src:
        problems.append("device selection anchor not found")
    if problems:
        print(f"[ERROR] {'; '.join(problems)}", file=sys.stderr)
        print("        The file differs from what this patch expects.",
              file=sys.stderr)
        return 3

    patched = src.replace(SIG_ANCHOR, SIG_PATCHED)
    patched = patched.replace(BODY_ANCHOR, BODY_PATCHED)

    # The patched source must parse, and learn_IVIM must gain exactly one
    # parameter with no other signature change.
    try:
        before = ast.parse(src)
        after = ast.parse(patched)
    except SyntaxError as e:
        print(f"[ERROR] patched source does not parse: {e}", file=sys.stderr)
        return 4

    def params(tree):
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "learn_IVIM")
        return [a.arg for a in fn.args.args]

    p_before, p_after = params(before), params(after)
    added = [p for p in p_after if p not in p_before]
    removed = [p for p in p_before if p not in p_after]

    print()
    print(" CHANGES")
    print(f"   signature: {len(p_before)} parameters -> {len(p_after)}")
    print(f"   added:     {added}")
    print(f"   removed:   {removed or 'none'}")
    print()
    print("   device selection becomes:")
    print("     explicit device argument, else cuda, else mps, else cpu")
    print("   CUDA and CPU behaviour is unchanged.")
    print()

    if removed or added != ["device"]:
        print("[ERROR] unexpected signature change, refusing to write",
              file=sys.stderr)
        return 5

    if not args.execute:
        print(" Re-run with --execute to apply.")
        print("=" * 70)
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{stamp}")
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")

    print(f" backup:  {backup.name}")
    print(f" written: {path.name}")
    print()
    print(" Verify:")
    print("   python -c \"import ast,inspect;"
          " from pace.learn_ivim import learn_IVIM;"
          " print('device' in inspect.signature(learn_IVIM).parameters)\"")
    print()
    print(" The backup is untracked. Remove it once you are satisfied,")
    print(" or revert with --revert.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
