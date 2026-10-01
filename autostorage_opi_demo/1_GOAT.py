"""Step 1: Conformer search for trans-pent-2-ene.

autostorage features: calculations linked to input/output geometries, property
values, and stationary points that are automatically tagged with identities
(InChI, SMILES, Hill formula, and the custom iRMSD conformer identity).
"""

import sys

from automol import rdkit_inchi
from autostorage import (
    CalculationGeometryLink,
    Database,
    PropertyValueRow,
    Role,
    StationaryPointRow,
    query,
)

import common
import orca

WORK_DIR = common.OUT_DIR / "1_GOAT"

with Database(common.DB_PATH, echo=common.ARGS.verbose) as db, db.session() as sess:
    # Identities make stored results easy to find, e.g. by InChI
    stmt = query.stationary_point_by_identity(rdkit_inchi, common.PENT2ENE_INCHI)
    if sess.exec(stmt).first() is not None:
        print("Pent-2-ene conformers are already stored.")
        sys.exit()

    xtb = common.get_or_create_model(sess, "XTB")
    hf3c = common.get_or_create_model(sess, "HF-3c")

    guess = common.from_smiles(common.PENT2ENE_SMILES)
    goat = orca.run(guess, xtb, "goat", WORK_DIR / "goat")
    sess.add(goat.calculation)
    for i, (conf, _) in enumerate(goat.frames("goat.finalensemble.xyz")):
        # Each conformer is an output geometry and a minimum (order 0)
        goat.calculation.geometry_links.append(
            CalculationGeometryLink(geometry=conf, role=Role.OUTPUT)
        )
        sess.add(StationaryPointRow(calculation=goat.calculation, geometry=conf))

        # Refine its energy with a more accurate model
        ene = orca.run(conf, hf3c, "Energy", WORK_DIR / f"ene_{i}")
        sess.add(
            PropertyValueRow(
                property_kind_name="energy",
                calculation=ene.calculation,
                geometry=conf,
                value=ene.energy(),
            )
        )

    sess.commit()
