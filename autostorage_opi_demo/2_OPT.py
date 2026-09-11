"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    Database,
    EnergyRow,
    GeometryRow,
    Role,
    StationaryPointRow,
)
from opi.input.structures import Properties, Structure

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
calc_input = CalculationInput(memory=5700, ncores=8)
calc_type = CalculationType.OPT

# Build the working directory
OPT_DIR = OUT_DIR / "2_OPT"
OPT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(DB_PATH, echo=False)

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

# Optimization of the pent2ene conformer
with db.session() as sess:
    # Query for existing wb97 model or create a new one :
    wb97_model = query.get_or_create_model(
        sess,
        program="ORCA",
        method="wb97x-3c",
        basis=None,
        program_version=ORCA_VERSION,
    )
    # Ensure xtb_model is in the current session
    sess.merge(wb97_model)

    min_conf = sess.get(GeometryRow, min_conf_id)
    if min_conf is None:
        msg = (
            f"Geometry with id {min_conf_id} could not be fetched from {DB_PATH}.",
            "\nPlease run 1_GOAT before running this script.",
        )
        raise KeyError(msg)

    # Query whether the calculation exists by checking if min_conf's InChI is tagged
    # to a calculation with model=wb97_model and calc_type="opt"
    calc_id = query.calculation(
        sess, model=wb97_model, calc_type=calc_type, geo=min_conf
    )

    if calc_id is not None:
        print(
            f"Pre-existing pent2ene OPT calculation found (id = {calc_id}).\n",
            "Skipping calculation.",
        )

    else:
        print("Beginning pent2ene OPT calculation.")
        calc_row, calc, output = run_calculation(
            pent2ene,
            work_dir=OPT_DIR,
            model=wb97_model,
            calc_type=calc_type,
            calc_input=calc_input,
        )

        # Link input Geometry to Calculation
        cg_link_in = CalculationGeometryLink(
            calculation=calc_row, geometry=min_conf, role=Role.INPUT
        )
