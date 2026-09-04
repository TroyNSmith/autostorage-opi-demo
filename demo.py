"""Demo for autostorage v0.0.13."""

import shutil
from pathlib import Path

from autostorage import CalculationRow, Database, EnergyRow, GeometryRow
from opi.input.structures import Structure
from sqlmodel import select

from helpers import calculation_exists, get_or_create_model, get_orca_version, run_goat

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

    ## Check if calculation already exists
    existing_calc = calculation_exists(sess, xtb_model, "goat", pent2ene)
    if existing_calc:
        print("GOAT calculation already exists, skipping...")
        ## Query for existing results
        stmt = select(EnergyRow).where(EnergyRow.calculation_id == existing_calc)
        energy_rows = [row for (row,) in sess.execute(stmt).all()]

        ## Get the lowest energy geometry from the existing calculation
        lowest_energy = min(energy_rows, key=lambda e: e.value)
        min_conf_id: int = lowest_energy.geometry_id

    else:
        goat_rows = run_goat(pent2ene, goat_dir, xtb_model)

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
with db.session() as sess:
    opt_dir = work_dir / "2_Optimization"
    opt_dir.mkdir(exist_ok=True, parents=True)

    ## Get or create the XTB model
    xtb_model = get_or_create_model(
        sess,
        program="orca",
        method="xtb",
        basis=None,
        program_version=get_orca_version(),
    )
    sess.merge(xtb_model)  # NOTE: Replace XTB with a different method/basis

    ## Get the lowest energy conformation from the GOAT calculation
    min_conf = sess.get(GeometryRow, min_conf_id)
    if min_conf is None:
        raise ValueError(f"Geometry with ID {min_conf_id} not found.")

    min_struc = Structure.from_xyz_block(
        min_conf.xyz_block(), charge=min_conf.charge, multiplicity=min_conf.spin - 1
    )
    raise ValueError(min_struc)
    existing_calc = calculation_exists(sess, xtb_model, "optimization", min_struc)
    if existing_calc:
        print("Optimization calculation already exists, skipping...")
    else:
        opt_rows = run_goat(
            min_conf, opt_dir, xtb_model
        )  # Assuming run_goat can be used for optimization
        sess.add_all(opt_rows)

    sess.flush()
    sess.commit()
    sess.close()
