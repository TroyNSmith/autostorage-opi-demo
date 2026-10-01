"""Step 2: Optimize the lowest-energy pent-2-ene conformer and the hydroxyl radical.

autostorage features: querying stationary points by identity and navigating
relationships (stationary point -> calculation, geometry -> property values).
"""

import sys

from automol import rdkit_inchi
from autostorage import Database, query

import common

WORK_DIR = common.OUT_DIR / "2_OPT"

with Database(common.DB_PATH, echo=common.ARGS.verbose) as db, db.session() as sess:
    stmt = query.stationary_point_by_identity(rdkit_inchi, common.PENT2ENE_INCHI)
    pent2ene = sess.exec(stmt).all()
    if any(stp.calculation.calc_type == "opt" for stp in pent2ene):
        print("Optimized pent-2-ene is already stored.")
        sys.exit()

    hf3c = common.get_or_create_model(sess, "HF-3c")

    # Pick the GOAT conformer with the lowest HF-3c energy
    conformers = [stp for stp in pent2ene if stp.calculation.calc_type == "goat"]
    lowest = min(conformers, key=lambda stp: common.energy(stp.geometry, hf3c))

    common.optimize(sess, lowest.geometry, hf3c, WORK_DIR / "pent2ene")
    hydroxyl = common.from_smiles(common.HYDROXYL_SMILES)
    common.optimize(sess, hydroxyl, hf3c, WORK_DIR / "hydroxyl")
    sess.commit()
