"""Demo for autostorage v0.0.13."""

import shutil
from pathlib import Path

from autostorage import Database, EnergyRow, ModelRow
from opi.input.structures import Structure

from helpers import get_orca_version, run_goat

# Clean up the working directory
work_dir = Path(__file__).resolve().parent / "out"
shutil.rmtree(work_dir, ignore_errors=True)
work_dir.mkdir(exist_ok=True, parents=True)

# Initialize the database
db_path = work_dir / "autostorage-demo.db"
db = Database(db_path)

# Build structures
pent2ene = Structure.from_smiles("CC=CCC")

with db.session() as sess:
    # 1. GOAT calculation (2-pentene)
    goat_dir = work_dir / "1_GOAT"
    goat_dir.mkdir(exist_ok=True, parents=True)

    ## Build the XTB model
    xtb_model = ModelRow(
        program="orca", method="xtb", basis=None, program_version=get_orca_version()
    )
    sess.add(xtb_model)

    ## Run the GOAT calculation
    goat_rows = run_goat(pent2ene, goat_dir, xtb_model)

    ## Add results to the session
    sess.add_all(goat_rows)

    ## Get the lowest value EnergyRow pending in the session
    energy_rows = [row for row in sess.new if isinstance(row, EnergyRow)]
    lowest_energy = min(energy_rows, key=lambda e: e.value)
    min_conf = lowest_energy.geometry  # Geometry corresponding to lowest energy

    sess.commit()
