"""Global Optimization of Pent-2-ene with the xTB model."""

import argparse

from autostorage import (
    CalculationGeometryLink,
    Database,
    EnergyRow,
    Role,
    StationaryPointRow,
)
from opi.input.structures import Properties, Structure

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

parser = argparse.ArgumentParser(
    prog="GOAT Demonstration",
    description="Run ORCA GOAT on pent2ene and store results in AutoStorage database.",
)
parser.add_argument(
    "-m",
    "--memory",
    help="Available memory in GB.",
    type=int,
    default=8,
)
parser.add_argument(
    "-n",
    "--ncores",
    help="Available number of CPU cores.",
    type=int,
    default=1,
)
parser.add_argument(
    "-v",
    "--verbose",
    help="Print SQL actions to terminal.",
    action="store_true",
)
args = parser.parse_args()

# Multiply memory by 0.75 as ORCA tends to bleed over alloc per documentation
mem_mib = int(args.memory * 953.7 * 0.75)

# Set calculation inputs
calc_input = CalculationInput(memory=mem_mib, ncores=args.ncores)
calc_type = CalculationType.GOAT

# Build the working directory
GOAT_DIR = OUT_DIR / "1_GOAT"
GOAT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(DB_PATH, echo=args.verbose)

# Build structure
pent2ene: Structure = Structure.from_smiles("CC=CCC")

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

    pent2ene_geo = structure_to_geometry(pent2ene)
    # Query whether the calculation exists by checking if pent2ene's InChI is tagged
    # to a calculation with model=xtb_model and calc_type="goat"
    calc_id = query.calculation(
        sess, model=xtb_model, calc_type=calc_type, geo=pent2ene_geo
    )

    if calc_id is not None:
        print(
            f"Pre-existing pent2ene GOAT calculation found (id = {calc_id}).\n",
            "Skipping calculation.",
        )

    else:
        print("Beginning pent2ene GOAT calculation.")
        calc_row, calc, _ = run_calculation(
            pent2ene,
            work_dir=GOAT_DIR,
            model=xtb_model,
            calc_type=calc_type,
            calc_input=calc_input,
        )
        # Link input Geometry to Calculation
        cg_link_in = CalculationGeometryLink(
            calculation=calc_row, geometry=pent2ene_geo, role=Role.INPUT
        )

        trjxyz = next(GOAT_DIR.glob("*.finalensemble.xyz"), None)
        if trjxyz is None:
            msg = f"finalensemble.xyz could not be located in {GOAT_DIR}."
            raise FileNotFoundError(msg)

        structures = Structure.from_trj_xyz(trjxyz)
        properties = Properties.from_trj_xyz(trjxyz, mode="goat")

        out_rows = [calc_row, pent2ene_geo, cg_link_in]
        for i, (struc, prop) in enumerate(
            zip(structures, properties, strict=True), start=1
        ):
            ene = prop.energy_total
            if ene is None:
                msg = f"Energy not found for Structure {i} in {trjxyz}."
                raise ValueError(msg)

            geo_row = structure_to_geometry(struc)
            ene_row = EnergyRow(calculation=calc_row, geometry=geo_row, value=ene)
            stp_row = StationaryPointRow(
                calculation=calc_row, geometry=geo_row, order=0
            )

            # Link output Geometry to Calculation
            cg_link_out = CalculationGeometryLink(
                calculation=calc_row, geometry=geo_row, role=Role.OUTPUT
            )
            out_rows.extend([geo_row, ene_row, stp_row, cg_link_out])

        sess.add_all(out_rows)
        sess.commit()
        sess.close()
