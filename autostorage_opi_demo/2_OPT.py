"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

import argparse

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
calc_type = CalculationType.OPT

# Build the working directory
OPT_DIR = OUT_DIR / "2_OPT"
OPT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(DB_PATH, echo=args.verbose)

# Build structures
pent2ene: Structure = Structure.from_smiles("CC=CCC")
hydroxyl: Structure = Structure.from_smiles("[OH]")


# Query whether the goat calculation exists by checking if pent2ene's InChI is tagged
# to a calculation with model=xtb_model and calc_type="goat". Then, fetch the geometry
# with the lowest energy for further optimization
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
    sess.merge(xtb_model)

    pent2ene_geo = structure_to_geometry(pent2ene)
    goat_id = query.calculation(
        sess, model=xtb_model, calc_type=CalculationType.GOAT, geo=pent2ene_geo
    )

    if goat_id is None:
        msg = (
            "Prerequisite pent2ene GOAT calculation not found.\n",
            "Please run 1_GOAT before running this script.",
        )
        raise KeyError(msg)

    print(f"Pre-existing pent2ene GOAT calculation found (id: {goat_id}).")
    goat_row = sess.get(CalculationRow, goat_id)
    if goat_row is None:
        msg = (
            f"Calculation with id {goat_id} could not be fetched from {DB_PATH}.",
            "\nPlease run 1_GOAT before running this script.",
        )
        raise KeyError(msg)

    sess.merge(goat_row)

    min_ene = min(goat_row.energies, key=lambda e: e.value)
    # Explicitly merge the minimum EnergyRow into this session
    min_conf_id = min_ene.geometry_id
    print(f"Lowest energy conformer identified (id = {min_conf_id})")


def optimize(db: Database, model: ModelRow, geo_in: GeometryRow) -> GeometryRow:
    """Optimize a geometry at model."""
    with db.session() as sess:
        model = sess.merge(model)
        geo_in = sess.merge(geo_in)
        # Query whether the calculation exists by checking if min_conf's InChI is tagged
        # to a calculation with model=wb97_model and calc_type="opt"
        calc_id = query.calculation(sess, model=model, calc_type=calc_type, geo=geo_in)

        if calc_id is not None:
            print(
                f"Pre-existing OPT calculation found (id = {calc_id}).\n",
                "Skipping calculation.",
            )
            calc_row = sess.get(CalculationRow, calc_id)
            if calc_row is None:
                msg = f"Error getting CalculationRow with {calc_id = }"
                raise LookupError(msg)

            cg_links = calc_row.geometry_links
            geo_out = [cg.geometry for cg in cg_links if cg.role == Role.OUTPUT]
            return geo_out[0]

        print("Beginning OPT calculation.")
        calc_row, _, output = run_calculation(
            pent2ene,
            work_dir=OPT_DIR,
            model=model,
            calc_type=calc_type,
            calc_input=calc_input,
        )
        # Link input Geometry to Calculation
        cg_link_in = CalculationGeometryLink(
            calculation=calc_row, geometry=pent2ene_min, role=Role.INPUT
        )
        sess.add_all([calc_row, geo_in, cg_link_in])

        struc = output.get_structure()
        grad = output.get_gradient(index=-2)  # Last gradient calculated
        ene = output.get_final_energy()

        if not struc or not grad or not ene:
            msg = "Optimization output did not return expected results."
            raise ValueError(msg)

        geo_out = structure_to_geometry(struc)
        ene_out = EnergyRow(calculation=calc_row, geometry=geo_out, value=ene)
        grad_out = GradientRow(calculation=calc_row, geometry=geo_out, value=grad)
        stp_out = StationaryPointRow(calculation=calc_row, geometry=geo_out, order=0)

        cg_link_out = CalculationGeometryLink(
            calculation=calc_row, geometry=geo_out, role=Role.OUTPUT
        )

        sess.add_all([geo_out, ene_out, grad_out, stp_out, cg_link_out])
        sess.commit()
        sess.close()

        return geo_out


# Get (or create) the models and pent2ene minimum conformer
with db.session() as sess:
    # Query for existing wb97 model or create a new one :
    wb97_model = query.get_or_create_model(
        sess,
        program="ORCA",
        method="wb97x-3c",
        basis=None,
        program_version=ORCA_VERSION,
    )

    pent2ene_min = sess.get(GeometryRow, min_conf_id)
    if pent2ene_min is None:
        msg = f"Error getting GeometryRow with {min_conf_id}."
        raise KeyError(msg)


# Optimize pent2ene
pent2ene_wb97 = optimize(db, wb97_model, pent2ene_min)
