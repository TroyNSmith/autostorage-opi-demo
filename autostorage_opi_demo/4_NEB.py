"""Step 4: Transition state search from the highest-energy scan point.

autostorage features: navigating from stored stationary points to their
trajectory and energies, validated stationary points (which require a stored
Hessian), and the reaction network (stages and steps).
"""

import sys

from automol import hill_formula
from autostorage import (
    CalculationGeometryLink,
    Database,
    PropertyValueRow,
    Role,
    StageRow,
    StationaryPointRow,
    StepRow,
    query,
)

import common
import orca

WORK_DIR = common.OUT_DIR / "4_NEB"

with Database(common.DB_PATH, echo=common.ARGS.verbose) as db, db.session() as sess:
    stmt = query.stationary_point_by_identity(
        hill_formula, common.COMPLEX_FORMULA, include_pseudo=True
    )
    stps = sess.exec(stmt).all()
    if any(stp.order == 1 for stp in stps):
        print("The transition state is already stored.")
        sys.exit()

    hf3c = common.get_or_create_model(sess, "HF-3c")

    # Follow the scan endpoints (step 3) back to the scan trajectory
    endpoints = [stp for stp in stps if stp.is_pseudo]
    trajectory = endpoints[0].calculation.trajectory_links[0].trajectory
    scan_geos = [link.geometry for link in trajectory.geometry_links]
    guess = max(scan_geos, key=lambda geo: common.energy(geo, hf3c))

    optts = orca.run(guess, hf3c, "optts", WORK_DIR / "optts")
    ts_geo = optts.geometry()
    optts.calculation.geometry_links.append(
        CalculationGeometryLink(geometry=ts_geo, role=Role.OUTPUT)
    )
    ts = StationaryPointRow(calculation=optts.calculation, geometry=ts_geo, order=1)
    sess.add(ts)
    common.validate(sess, ts, hf3c, WORK_DIR / "freq")

    ene = orca.run(ts_geo, hf3c, "Energy", WORK_DIR / "ene")
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
