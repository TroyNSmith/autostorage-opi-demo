# autostorage-opi-demo

A worked example of [`autostorage`](https://github.com/TroyNSmith/autostorage): every result of a small
reaction study (H-abstraction from pent-2-ene by OH) is stored in a single
SQLite database, connected by provenance and found again by chemical identity.
ORCA (via OPI) does the quantum chemistry, but is kept out of the way in
[`orca.py`](autostorage_opi_demo/orca.py).

## Steps

| Script | Calculation | autostorage features |
|---|---|---|
| [`1_GOAT.py`](autostorage_opi_demo/1_GOAT.py) | Conformer search (XTB) + energies (HF-3c) | Calculations with input/output geometries, property values, stationary points with automatic identities |
| [`2_OPT.py`](autostorage_opi_demo/2_OPT.py) | Optimize lowest conformer and OH | `query.stationary_point_by_identity`, relationship navigation |
| [`3_SCAN.py`](autostorage_opi_demo/3_SCAN.py) | Relaxed scan of the abstraction coordinate | Trajectories, pseudo stationary points |
| [`4_NEB.py`](autostorage_opi_demo/4_NEB.py) | TS search from the highest scan point | Validated stationary points (Hessian required), stages and steps |
| [`5_IRC.py`](autostorage_opi_demo/5_IRC.py) | IRC and endpoint optimization | Step validations, shared identity rows, updating the reaction network |

Each script skips itself if its results are already stored, so the pipeline
can be rerun safely.

Supporting modules:

- [`extensions.py`](autostorage_opi_demo/extensions.py): registers a custom
  identity algorithm (iRMSD conformer grouping) and property kind (`zpe`),
  which autostorage then handles like the built-in ones.
- [`common.py`](autostorage_opi_demo/common.py): shared constants and
  autostorage recipes (`optimize`, `validate`, ...).
- [`orca.py`](autostorage_opi_demo/orca.py): runs ORCA and parses its output.

## Running

```bash
uv sync
uv run python autostorage_opi_demo/1_GOAT.py -m 8 -n 8  # and so on for 2-5
```

or submit [`submit.sh`](submit.sh) to SLURM. Results are written to `out/`
(`out/demo.db` is the database). Finished ORCA calculations in `out/` are
reused.
