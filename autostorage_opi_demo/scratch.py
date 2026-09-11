"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    Database,
    EnergyRow,
    GeometryRow,
    GradientRow,
    ModelRow,
    Role,
    StationaryPointRow,
)
from opi.input.structures import Structure
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import conf_ident  # noqa: F401
import query
from utils import (
    DB_PATH,
    ORCA_VERSION,
    OUT_DIR,
    CalculationInput,
    CalculationType,
    run_calculation,
    structure_to_geometry,
)

# Set calculation inputs
calc_input = CalculationInput(memory=5700, ncores=1)
calc_type = CalculationType.OPT

# Build the working directory
OPT_DIR = OUT_DIR / "2_OPT"
OPT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(DB_PATH, echo=False)

# Build structures
mol1 = Chem.MolFromSmiles("CCCCCCC")
mol1 = Chem.AddHs(mol1)
rdDistGeom.EmbedMolecule(mol1)
xyz1 = Chem.MolToXYZBlock(mol1)

mol2 = Chem.MolFromSmiles("CCCCCCC")
mol2 = Chem.AddHs(mol2)
rdDistGeom.EmbedMolecule(mol2)
xyz2 = Chem.MolToXYZBlock(mol2)

hydroxyl1: Structure = Structure.from_xyz_block(xyz1)
hydroxyl2: Structure = Structure.from_xyz_block(xyz2)

for hydroxyl in [hydroxyl1, hydroxyl2]:
    with db.session() as sess:
        # Query for existing xtb model or create a new one :
        xtb_model = query.get_or_create_model(
            sess,
            program="ORCA",
            method="xtb",
            basis=None,
            program_version=ORCA_VERSION,
        )
        # Ensure xtb_model is in the current session
        sess.add(xtb_model)

        print("Beginning OPT calculation.")
        calc_row, _, output = run_calculation(
            hydroxyl,
            work_dir=OPT_DIR,
            model=xtb_model,
            calc_type=calc_type,
            calc_input=calc_input,
        )
        # Link input Geometry to Calculation
        geo = structure_to_geometry(hydroxyl)
        cg_link_in = CalculationGeometryLink(
            calculation=calc_row, geometry=geo, role=Role.INPUT
        )
        sess.add_all([calc_row, geo, cg_link_in])

        struc = output.get_structure()

        if not struc:
            msg = "Optimization output did not return expected results."
            raise ValueError(msg)

        geo_out = structure_to_geometry(struc)
        stp_out = StationaryPointRow(calculation=calc_row, geometry=geo_out, order=0)

        cg_link_out = CalculationGeometryLink(
            calculation=calc_row, geometry=geo_out, role=Role.OUTPUT
        )

        sess.add_all([geo_out, stp_out, cg_link_out])
        sess.commit()
        sess.close()
