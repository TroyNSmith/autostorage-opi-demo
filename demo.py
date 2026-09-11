"""Demo for autostorage v0.0.13."""

import shutil
from pathlib import Path

from autostorage import CalculationRow, Database, EnergyRow, GeometryRow
from opi.input.structures import Structure
from sqlmodel import select

import helpers
from helpers import calculation_exists, get_or_create_model, get_orca_version, run_calc

# Clean up the working directory
work_dir = Path(__file__).resolve().parent / "out"
work_dir.mkdir(exist_ok=True, parents=True)

# Initialize the database
db_path = work_dir / "autostorage-demo.db"
db = Database(db_path, echo=False)

# Build structures
pent2ene = Structure.from_smiles("CC=CCC")
hydroxyl = Structure.from_smiles("[OH]")

# 1. GOAT calculation (2-pentene)
with db.session() as sess:
    goat_dir = work_dir / "1_GOAT"
    goat_dir.mkdir(exist_ok=True, parents=True)

    ## Get or create the XTB model
    xtb_model = get_or_create_model(
        sess,
        program="orca",
        method="xtb",
        basis=None,
        program_version=get_orca_version(),
    )
    sess.merge(xtb_model)

    pent2ene_geo = GeometryRow.from_xyz_block(
        pent2ene.to_xyz_block(), charge=pent2ene.charge, spin=pent2ene.multiplicity + 1
    )

    ## Check if calculation already exists
    existing_calc = calculation_exists(sess, xtb_model, "goat", pent2ene_geo)
    if existing_calc:
        print("GOAT calculation already exists, skipping...")
        ## Query for existing results
        stmt = select(EnergyRow).where(EnergyRow.calculation_id == existing_calc)
        energy_rows = [row for (row,) in sess.execute(stmt).all()]

        ## Get the lowest energy geometry from the existing calculation
        lowest_energy = min(energy_rows, key=lambda e: e.value)
        min_conf_id: int = lowest_energy.geometry_id

    else:
        goat_rows = helpers.goat(pent2ene_geo, goat_dir, xtb_model)

        ## Add results to the session
        sess.add_all(goat_rows)

        ## Get the lowest value EnergyRow pending in the session
        energy_rows = [row for row in sess.new if isinstance(row, EnergyRow)]
        lowest_energy = min(energy_rows, key=lambda e: e.value)
        min_conf_id: int = (
            lowest_energy.geometry_id
        )  # Geometry corresponding to lowest energy

    sess.flush()
    sess.commit()
    sess.close()

# 2. Optimization of hydroxyl and the lowest energy geometry from the GOAT calculation
opt_dir = work_dir / "2_Optimization"
opt_dir.mkdir(exist_ok=True, parents=True)

## a. pent2ene lowest energy conformer
with db.session() as sess:
    wb97x3c_model = get_or_create_model(
        sess,
        program="orca",
        method="wb97x-3c",
        basis=None,
        program_version=get_orca_version(),
    )
    sess.merge(wb97x3c_model)

    ## Get the lowest energy conformation from the GOAT calculation
    min_conf = sess.get(GeometryRow, min_conf_id)
    if min_conf is None:
        raise ValueError(f"Geometry with ID {min_conf_id} not found.")

    existing_calc = calculation_exists(sess, wb97x3c_model, "optimization", min_conf)
    if existing_calc:
        print("Optimization calculation already exists, skipping...")
    else:
        opt_rows = helpers.optimization(min_conf, opt_dir / "pent2ene", wb97x3c_model)
        sess.add_all(opt_rows)

    sess.flush()
    sess.commit()
    sess.close()

## b. hydroxyl
with db.session() as sess:
    wb97x3c_model = get_or_create_model(
        sess,
        program="orca",
        method="wb97x-3c",
        basis=None,
        program_version=get_orca_version(),
    )
    sess.merge(wb97x3c_model)

    hydroxyl_geo = GeometryRow.from_xyz_block(
        hydroxyl.to_xyz_block(), charge=hydroxyl.charge, spin=hydroxyl.multiplicity + 1
    )

    existing_calc = calculation_exists(
        sess, wb97x3c_model, "optimization", hydroxyl_geo
    )
    if existing_calc:
        print("Optimization calculation already exists, skipping...")
    else:
        opt_rows = helpers.optimization(
            hydroxyl_geo, opt_dir / "hydroxyl", wb97x3c_model
        )
        sess.add_all(opt_rows)

    sess.flush()
    sess.commit()
    sess.close()
