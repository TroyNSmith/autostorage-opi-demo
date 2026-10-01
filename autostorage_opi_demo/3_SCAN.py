"""Step 3: Relaxed scan of H-abstraction from pent-2-ene by OH.

autostorage features: trajectories (ordered geometries linked to a
calculation) and pseudo stationary points, which queries skip by default.
"""

import sys

from automol import hill_formula, rdkit_inchi
from autostorage import (
    CalculationGeometryLink,
    CalculationTrajectoryLink,
    Database,
    GeometryRow,
    GeometryTrajectoryLink,
    PropertyValueRow,
    Role,
    StationaryPointRow,
    TrajectoryRow,
    query,
)

import common
import orca

WORK_DIR = common.OUT_DIR / "3_SCAN"

with Database(common.DB_PATH, echo=common.ARGS.verbose) as db, db.session() as sess:
    stmt = query.stationary_point_by_identity(
        hill_formula, common.COMPLEX_FORMULA, include_pseudo=True
    )
    if sess.exec(stmt).first() is not None:
        print("The pent-2-ene + OH scan is already stored.")
        sys.exit()

    xtb = common.get_or_create_model(sess, "XTB")
    hf3c = common.get_or_create_model(sess, "HF-3c")

    def optimized(inchi: str) -> GeometryRow:
        """Get the geometry of an optimized minimum by InChI."""
        stmt = query.stationary_point_by_identity(rdkit_inchi, inchi)
        return next(
            stp.geometry
            for stp in sess.exec(stmt)
            if stp.calculation.calc_type == "opt"
        )

    # Place OH 1.4 Å from the allylic H (atom 10) of pent-2-ene
    pent2ene = optimized(common.PENT2ENE_INCHI)
    complex_geo = common.form_complex(
        pent2ene, optimized(common.HYDROXYL_INCHI), 10, 0, 1.4
    )

    # Relax the complex with the reacting C, H, and O atoms frozen
    const = orca.run(
        complex_geo,
        xtb,
        "opt",
        WORK_DIR / "constrained_opt",
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
        WORK_DIR / "scan",
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
        ene = orca.run(geo, hf3c, "Energy", WORK_DIR / f"ene_{i}")
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
