# autostorage-opi-demo

A worked example of [`autostorage`](https://github.com/TroyNSmith/autostorage): every result of a small
reaction study (H-abstraction from pent-2-ene by OH) is stored in a single
SQLite database, connected by provenance and found again by chemical identity.
ORCA (via OPI) does the quantum chemistry, but is kept out of the way in
[`orca.py`](autostorage_opi_demo/orca.py).

## Steps

All steps live in one script, [`demo.py`](autostorage_opi_demo/demo.py),
split into `# %%` cells. After each step, a cell visualizes its stored results
with the viewers built into automol (`geom.view`, `geom.render_svg`) and
autostorage (`TrajectoryRow.view`).

| Step | Calculation | autostorage features | Visualization |
|---|---|---|---|
| 1 | Conformer search (XTB) + energies (HF-3c) | Calculations with input/output geometries, property values, stationary points with automatic identities | Conformers with relative energies |
| 2 | Optimize lowest conformer and OH | `query.stationary_point_by_identity`, relationship navigation | Optimized reactants (atom-labeled) |
| 3 | Relaxed scan of the abstraction coordinate | Trajectories, pseudo stationary points | Animated scan trajectory |
| 4 | TS search from the highest scan point | Validated stationary points (Hessian required), stages and steps | Transition state (atom-labeled) |
| 5 | IRC and endpoint optimization | Step validations, shared identity rows, updating the reaction network | Animated IRC branches, reaction step energies |

Each step skips itself if its results are already stored, so the pipeline
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
uv run python autostorage_opi_demo/demo.py -m 8 -n 8
```

or run it cell by cell in Jupyter (e.g. the VS Code interactive window) to see
the visualizations, or submit [`submit.sh`](submit.sh) to SLURM. Results are
written to `out/` (`out/demo.db` is the database). Finished ORCA calculations
in `out/` are reused.
