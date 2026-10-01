"""Step 5: Validate the reaction step with an IRC and refine its endpoints.

autostorage features: validations attached to reaction steps, shared identity
rows (including the custom iRMSD conformer identity), and updating the reaction
network.
"""

import sys

from automol import hill_formula
from autostorage import (
    CalculationTrajectoryLink,
    Database,
    GeometryTrajectoryLink,
    PropertyValueRow,
    Role,
    StationaryPointRow,
    TrajectoryRow,
    ValidationRow,
    query,
)

import common
import orca

WORK_DIR = common.OUT_DIR / "5_IRC"

with Database(common.DB_PATH, echo=common.ARGS.verbose) as db, db.session() as sess:
    stmt = query.stationary_point_by_identity(hill_formula, common.COMPLEX_FORMULA)
    ts = next(stp for stp in sess.exec(stmt) if stp.order == 1)
    step = ts.stages[0].steps[0]
    if step.validations:
        print("The reaction step is already validated.")
        sys.exit()

    hf3c = common.get_or_create_model(sess, "HF-3c")

    irc = orca.run(ts.geometry, hf3c, "IRC", WORK_DIR / "irc")
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
        minimum = common.optimize(sess, lowest, hf3c, WORK_DIR / f"opt_{i}")
        common.validate(sess, minimum, hf3c, WORK_DIR / f"freq_{i}")
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
