"""The autostorage demo, run as a script or cell by cell in Jupyter."""

# %% [markdown]
# # autostorage + ORCA demo: H-abstraction from pent-2-ene by OH
#
# Every result is stored in a single SQLite database (`out/demo.db`), connected
# by provenance and found again by chemical identity. Each step skips itself if
# its results are already stored, so the cells (or the whole script) can be
# rerun safely. Visualization cells only display output in Jupyter (e.g. the
# VS Code interactive window).

# %%
import sys
from pathlib import Path

import py3Dmol
from automol import geom, hill_formula, rdkit_inchi
from autostorage import (
    AutostorageSession,
    CalculationGeometryLink,
    CalculationTrajectoryLink,
    Database,
    GeometryRow,
    GeometryTrajectoryLink,
    PropertyValueRow,
    Role,
    StageRow,
    StationaryPointRow,
    StepRow,
    TrajectoryRow,
    ValidationRow,
    query,
)
from IPython import get_ipython
from IPython.display import display

# Make the sibling modules importable from any working directory (the VS Code
# interactive window sets `__file__`, but may run elsewhere)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import orca

HARTREE_TO_KCAL = 627.509474

db = Database(common.DB_PATH, echo=common.ARGS.verbose)


def show(*objs: object) -> None:
    """Display objects in Jupyter; skipped when run as a plain script."""
    if get_ipython() is None:
        return
    for obj in objs:
        # Displaying a py3Dmol view directly would also print its text repr
        if isinstance(obj, py3Dmol.view):
            obj.show()
        else:
            display(obj)


# %% [markdown]
# ## Step 1: Conformer search for trans-pent-2-ene
#
# autostorage features: calculations linked to input/output geometries,
# property values, and stationary points that are automatically tagged with
# identities (InChI, SMILES, Hill formula, and the custom iRMSD conformer
# identity).


# %%
def step1_goat(sess: AutostorageSession) -> None:
    """Run a GOAT conformer search and refine the conformer energies."""
    work_dir = common.OUT_DIR / "1_GOAT"

    # Identities make stored results easy to find, e.g. by InChI
    stmt = query.stationary_point_by_identity(rdkit_inchi, common.PENT2ENE_INCHI)
    if sess.exec(stmt).first() is not None:
        print("Pent-2-ene conformers are already stored.")
        return

    xtb = common.get_or_create_model(sess, "XTB")
    hf3c = common.get_or_create_model(sess, "HF-3c")

    guess = common.from_smiles(common.PENT2ENE_SMILES)
    goat = orca.run(guess, xtb, "goat", work_dir / "goat")
    sess.add(goat.calculation)
    for i, (conf, _) in enumerate(goat.frames("goat.finalensemble.xyz")):
        # Each conformer is an output geometry and a minimum (order 0)
        goat.calculation.geometry_links.append(
            CalculationGeometryLink(geometry=conf, role=Role.OUTPUT)
        )
        sess.add(StationaryPointRow(calculation=goat.calculation, geometry=conf))

        # Refine its energy with a more accurate model
        ene = orca.run(conf, hf3c, "Energy", work_dir / f"ene_{i}")
        sess.add(
            PropertyValueRow(
                property_kind_name="energy",
                calculation=ene.calculation,
                geometry=conf,
                value=ene.energy(),
            )
        )

    sess.commit()


with db.session() as sess:
    step1_goat(sess)

# %%
# The GOAT conformers, with their HF-3c energies relative to the lowest one
with db.session() as sess:
    hf3c = common.get_or_create_model(sess, "HF-3c")
    stmt = query.stationary_point_by_identity(rdkit_inchi, common.PENT2ENE_INCHI)
    conformers = [stp for stp in sess.exec(stmt) if stp.calculation.calc_type == "goat"]
    energies = [common.energy(stp.geometry, hf3c) for stp in conformers]
    for stp, ene in sorted(zip(conformers, energies, strict=True), key=lambda x: x[1]):
        rel = (ene - min(energies)) * HARTREE_TO_KCAL
        print(f"Conformer {stp.id}: {rel:.2f} kcal/mol")
        show(geom.render_svg(stp.geometry))

# %% [markdown]
# ## Step 2: Optimize the lowest-energy conformer and the hydroxyl radical
#
# autostorage features: querying stationary points by identity and navigating
# relationships (stationary point -> calculation, geometry -> property values).


# %%
def step2_opt(sess: AutostorageSession) -> None:
    """Optimize the lowest-energy pent-2-ene conformer and the hydroxyl radical."""
    work_dir = common.OUT_DIR / "2_OPT"

    stmt = query.stationary_point_by_identity(rdkit_inchi, common.PENT2ENE_INCHI)
    pent2ene = sess.exec(stmt).all()
    if any(stp.calculation.calc_type == "opt" for stp in pent2ene):
        print("Optimized pent-2-ene is already stored.")
        return

    hf3c = common.get_or_create_model(sess, "HF-3c")

    # Pick the GOAT conformer with the lowest HF-3c energy
    conformers = [stp for stp in pent2ene if stp.calculation.calc_type == "goat"]
    lowest = min(conformers, key=lambda stp: common.energy(stp.geometry, hf3c))

    common.optimize(sess, lowest.geometry, hf3c, work_dir / "pent2ene")
    hydroxyl = common.from_smiles(common.HYDROXYL_SMILES)
    common.optimize(sess, hydroxyl, hf3c, work_dir / "hydroxyl")
    sess.commit()


with db.session() as sess:
    step2_opt(sess)


# %%
def optimized(sess: AutostorageSession, inchi: str) -> GeometryRow:
    """Get the geometry of an optimized minimum by InChI."""
    stmt = query.stationary_point_by_identity(rdkit_inchi, inchi)
    return next(
        stp.geometry for stp in sess.exec(stmt) if stp.calculation.calc_type == "opt"
    )


# The optimized reactants, labeled by atom index (H 10 is abstracted in step 3)
with db.session() as sess:
    show(
        geom.view(optimized(sess, common.PENT2ENE_INCHI), label=True),
        geom.view(optimized(sess, common.HYDROXYL_INCHI), label=True),
    )

# %% [markdown]
# ## Step 3: Relaxed scan of H-abstraction from pent-2-ene by OH
#
# autostorage features: trajectories (ordered geometries linked to a
# calculation) and pseudo stationary points, which queries skip by default.


# %%
def step3_scan(sess: AutostorageSession) -> None:
    """Scan the forming O-H bond of the pent-2-ene + OH complex."""
    work_dir = common.OUT_DIR / "3_SCAN"

    stmt = query.stationary_point_by_identity(
        hill_formula, common.COMPLEX_FORMULA, include_pseudo=True
    )
    if sess.exec(stmt).first() is not None:
        print("The pent-2-ene + OH scan is already stored.")
        return

    xtb = common.get_or_create_model(sess, "XTB")
    hf3c = common.get_or_create_model(sess, "HF-3c")

    # Place OH 1.4 Å from the allylic H (atom 10) of pent-2-ene
    pent2ene = optimized(sess, common.PENT2ENE_INCHI)
    complex_geo = common.form_complex(
        pent2ene, optimized(sess, common.HYDROXYL_INCHI), 10, 0, 1.4
    )

    # Relax the complex with the reacting C, H, and O atoms frozen
    const = orca.run(
        complex_geo,
        xtb,
        "opt",
        work_dir / "constrained_opt",
        "%geom constraints {C 3 C} {C 10 C} {C 15 C} end end",
    )
    start = const.geometry()
    const.calculation.geometry_links.append(
        CalculationGeometryLink(geometry=start, role=Role.OUTPUT)
    )
    sess.add(const.calculation)

    # Scan the forming O-H bond and store the scan points as a trajectory
    scan = orca.run(
        start,
        xtb,
        "opt",
        work_dir / "scan",
        "%geom scan B 15 10 = 1.4, 1.0, 10 end end",
    )
    trajectory = TrajectoryRow()
    scan.calculation.trajectory_links.append(
        CalculationTrajectoryLink(trajectory=trajectory, role=Role.OUTPUT)
    )
    sess.add(scan.calculation)

    frames = [geo for geo, _ in scan.frames("opt.allxyz")]
    for i, geo in enumerate(frames):
        trajectory.geometry_links.append(
            GeometryTrajectoryLink(geometry=geo, index=[i])
        )
        ene = orca.run(geo, hf3c, "Energy", work_dir / f"ene_{i}")
        sess.add(
            PropertyValueRow(
                property_kind_name="energy",
                calculation=ene.calculation,
                geometry=geo,
                value=ene.energy(),
            )
        )

    # The scan endpoints are initial guesses for the reactant and product complexes
    for geo in (frames[0], frames[-1]):
        sess.add(
            StationaryPointRow(
                calculation=scan.calculation, geometry=geo, is_pseudo=True
            )
        )

    sess.commit()


with db.session() as sess:
    step3_scan(sess)

# %%
# The scan trajectory, found from its calculation's output trajectory link
with db.session() as sess:
    hf3c = common.get_or_create_model(sess, "HF-3c")
    stmt = query.stationary_point_by_identity(
        hill_formula, common.COMPLEX_FORMULA, include_pseudo=True
    )
    scan_calc = next(stp.calculation for stp in sess.exec(stmt) if stp.is_pseudo)
    trajectory = scan_calc.trajectory_links[0].trajectory
    scan_geos = trajectory.geometries_along()
    energies = [common.energy(geo, hf3c) for geo in scan_geos]
    for i, ene in enumerate(energies):
        print(f"Scan point {i}: {(ene - energies[0]) * HARTREE_TO_KCAL:.2f} kcal/mol")
    show(trajectory.view())

# %% [markdown]
# ## Step 4: Transition state search from the highest-energy scan point
#
# autostorage features: navigating from stored stationary points to their
# trajectory and energies, validated stationary points (which require a stored
# Hessian), and the reaction network (stages and steps).


# %%
def step4_ts(sess: AutostorageSession) -> None:
    """Optimize a TS from the highest scan point and add the reaction step."""
    work_dir = common.OUT_DIR / "4_NEB"

    stmt = query.stationary_point_by_identity(
        hill_formula, common.COMPLEX_FORMULA, include_pseudo=True
    )
    stps = sess.exec(stmt).all()
    if any(stp.order == 1 for stp in stps):
        print("The transition state is already stored.")
        return

    hf3c = common.get_or_create_model(sess, "HF-3c")

    # Follow the scan endpoints (step 3) back to the scan trajectory
    endpoints = [stp for stp in stps if stp.is_pseudo]
    trajectory = endpoints[0].calculation.trajectory_links[0].trajectory
    scan_geos = [link.geometry for link in trajectory.geometry_links]
    guess = max(scan_geos, key=lambda geo: common.energy(geo, hf3c))

    optts = orca.run(guess, hf3c, "optts", work_dir / "optts")
    ts_geo = optts.geometry()
    optts.calculation.geometry_links.append(
        CalculationGeometryLink(geometry=ts_geo, role=Role.OUTPUT)
    )
    ts = StationaryPointRow(calculation=optts.calculation, geometry=ts_geo, order=1)
    sess.add(ts)
    common.validate(sess, ts, hf3c, work_dir / "freq")

    ene = orca.run(ts_geo, hf3c, "Energy", work_dir / "ene")
    sess.add(
        PropertyValueRow(
            property_kind_name="energy",
            calculation=ene.calculation,
            geometry=ts_geo,
            value=ene.energy(),
        )
    )

    # Connect the scan endpoints through the TS as an elementary reaction step
    sess.add(
        StepRow(
            stage1=StageRow(stationaries=[endpoints[0]]),
            stage2=StageRow(stationaries=[endpoints[1]]),
            stage_ts=StageRow(stationaries=[ts], is_ts=True),
        )
    )
    sess.commit()


with db.session() as sess:
    step4_ts(sess)


# %%
def transition_state(sess: AutostorageSession) -> StationaryPointRow:
    """Get the stored transition state (order 1) of the complex."""
    stmt = query.stationary_point_by_identity(hill_formula, common.COMPLEX_FORMULA)
    return next(stp for stp in sess.exec(stmt) if stp.order == 1)


# The transition state, labeled by atom index (C 3 ... H 10 ... O 15)
with db.session() as sess:
    show(geom.view(transition_state(sess).geometry, label=True))

# %% [markdown]
# ## Step 5: Validate the reaction step with an IRC and refine its endpoints
#
# autostorage features: validations attached to reaction steps, shared identity
# rows (including the custom iRMSD conformer identity), and updating the
# reaction network.


# %%
def step5_irc(sess: AutostorageSession) -> None:
    """Run an IRC from the TS and replace the step's scan guesses with its minima."""
    work_dir = common.OUT_DIR / "5_IRC"

    ts = transition_state(sess)
    step = ts.stages[0].steps[0]
    if step.validations:
        print("The reaction step is already validated.")
        return

    hf3c = common.get_or_create_model(sess, "HF-3c")

    irc = orca.run(ts.geometry, hf3c, "IRC", work_dir / "irc")
    sess.add(ValidationRow(calculation=irc.calculation, step=step, method="irc"))

    minima: list[StationaryPointRow] = []
    for i, direction in enumerate("BF"):
        # Store each IRC branch as a trajectory with its energies
        trajectory = TrajectoryRow()
        irc.calculation.trajectory_links.append(
            CalculationTrajectoryLink(trajectory=trajectory, role=Role.OUTPUT)
        )
        frames = irc.frames(f"IRC_IRC_{direction}_trj.xyz")
        for j, (geo, energy) in enumerate(frames):
            trajectory.geometry_links.append(
                GeometryTrajectoryLink(geometry=geo, index=[j])
            )
            sess.add(
                PropertyValueRow(
                    property_kind_name="energy",
                    calculation=irc.calculation,
                    geometry=geo,
                    value=energy,
                )
            )

        # Optimize the lowest point of the branch into a validated minimum
        lowest, _ = min(frames, key=lambda frame: frame[1])
        minimum = common.optimize(sess, lowest, hf3c, work_dir / f"opt_{i}")
        common.validate(sess, minimum, hf3c, work_dir / f"freq_{i}")
        minima.append(minimum)

    # Flush to generate identities for the new minima. Identity rows are shared,
    # so pair each stage's scan guess with the minimum it shares the most with.
    sess.flush()
    stages = [step.stage1, step.stage2]
    guesses = [stage.stationaries[0] for stage in stages]

    def shared(a: StationaryPointRow, b: StationaryPointRow) -> set[str]:
        """Names of the identity algorithms whose values two points share."""
        ids = {i.id for i in b.identities}
        return {i.algorithm.name for i in a.identities if i.id in ids}

    def score(pairs: list[StationaryPointRow]) -> int:
        """Count the identities shared by the guesses and `pairs`."""
        return sum(len(shared(g, m)) for g, m in zip(guesses, pairs, strict=True))

    if score(minima[::-1]) > score(minima):
        minima.reverse()

    for stage, guess, minimum in zip(stages, guesses, minima, strict=True):
        print(f"Stage {stage.id}: replacing scan guess with IRC minimum")
        print(f"  shared identities: {sorted(shared(guess, minimum))}")
        stage.stationaries = [minimum]

    sess.commit()


with db.session() as sess:
    step5_irc(sess)

# %%
# The IRC branches, found from the validation attached to the reaction step
with db.session() as sess:
    step = transition_state(sess).stages[0].steps[0]
    irc_calc = step.validations[0].calculation
    for i, link in enumerate(irc_calc.trajectory_links):
        print(f"IRC branch {i} ({len(link.trajectory.geometry_links)} points)")
        show(link.trajectory.view())

# %%
# The validated reaction step: stage 1, TS, and stage 2, with HF-3c energies
# (and zero-point corrected energies) relative to stage 1
with db.session() as sess:
    hf3c = common.get_or_create_model(sess, "HF-3c")
    step = transition_state(sess).stages[0].steps[0]
    stages = {
        f"Stage {step.stage1.id}": step.stage1,
        "TS": step.stage_ts,
        f"Stage {step.stage2.id}": step.stage2,
    }
    ref_ene = ref_zpe = None
    for name, stage in stages.items():
        geo = stage.stationaries[0].geometry
        ene = common.energy(geo, hf3c)
        zpe = next(
            float(prop.value)
            for prop in geo.properties
            if prop.property_kind_name == "zpe"
        )
        if ref_ene is None:
            ref_ene, ref_zpe = ene, ene + zpe
        rel_ene = (ene - ref_ene) * HARTREE_TO_KCAL
        rel_zpe = (ene + zpe - ref_zpe) * HARTREE_TO_KCAL
        print(f"{name}: {rel_ene:.2f} kcal/mol ({rel_zpe:.2f} kcal/mol with ZPE)")
        show(geom.render_svg(geo))

# %%
db.close()
