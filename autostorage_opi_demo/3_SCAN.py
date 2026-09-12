"""Optimize hydroxyl-pent2ene complex then scan rxn coord to find highest energy."""

import argparse
from pathlib import Path

import numpy as np
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
from opi.input.blocks import BlockGeom, Constraint
from opi.input.structures import Structure

import query
from utils import (
    DB_PATH,
    ORCA_VERSION,
    OUT_DIR,
    CalculationInput,
    CalculationType,
    ModelKeywords,
    geo_to_struc,
    get_logger,
    run_calculation,
    struc_to_geo,
)

logger = get_logger(__name__)

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
SCAN_DIR = OUT_DIR / "3_SCAN"
SCAN_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(DB_PATH, echo=args.verbose)

# Build structures
pent2ene: Structure = Structure.from_smiles("CC=CCC")
hydroxyl: Structure = Structure.from_smiles("[OH]")

with db.session() as sess:
    # Query for existing xtb model or create a new one :
    revdsd_model = query.get_or_create_model(
        sess,
        program="ORCA",
        method="revDSD-PBEP86-D4/2021",
        basis="DEF2-QZVPP",
        program_version=ORCA_VERSION,
        keywords=ModelKeywords(corrections=["DEF2-QZVPP/C"]),
    )
    # Ensure revdsd_model is in the current session
    sess.add(revdsd_model)

    pent2ene_geo = struc_to_geo(pent2ene)
    pent2ene_calc_id = query.calculation(
        sess, model=revdsd_model, calc_type=CalculationType.OPT, geo=pent2ene_geo
    )
    pent2ene_calc = sess.get(CalculationRow, pent2ene_calc_id)

    if pent2ene_calc is None:
        msg = f"Calculation with id {pent2ene_calc_id} could not be fetched from {DB_PATH}."
        raise KeyError(msg)

    pent2ene_revd = next(
        cg.geometry for cg in pent2ene_calc.geometry_links if cg.role == Role.OUTPUT
    )

    hydroxyl_geo = struc_to_geo(hydroxyl)
    hydroxyl_calc_id = query.calculation(
        sess, model=revdsd_model, calc_type=CalculationType.OPT, geo=hydroxyl_geo
    )
    hydroxyl_calc = sess.get(CalculationRow, hydroxyl_calc_id)

    if hydroxyl_calc is None:
        msg = f"Calculation with id {hydroxyl_calc_id} could not be fetched from {DB_PATH}."
        raise KeyError(msg)

    hydroxyl_revd = next(
        cg.geometry for cg in hydroxyl_calc.geometry_links if cg.role == Role.OUTPUT
    )

# Build the [OH].C(C)C=CC complex
pent2ene_xyzs = pent2ene_revd.coordinates
pent2ene_syms = pent2ene_revd.symbols
hydroxyl_xyzs = hydroxyl_revd.coordinates
hydroxyl_syms = hydroxyl_revd.symbols

# Place the hydroxyl fragment for the H-abstraction pre-reactive complex: OH
# abstracts the allylic hydrogen H11 (index 10, bound to the CH2 at index 3).
# The hydroxyl fragment is only translated/rotated as a rigid body, so its
# O16-H17 bond length is untouched, and none of pent2ene's internal
# coordinates are touched.
O_H11_DISTANCE = 1.15  # Target O16-H11 distance
H11_IDX = 10

h11_xyz = pent2ene_xyzs[H11_IDX]

# The normal to the carbon backbone plane gives the "perpendicular to
# pent2ene" direction; approaching along it (rather than in-plane) keeps the
# hydroxyl fragment from overlapping with the rest of the molecule.
carbon_xyzs = pent2ene_xyzs[[i for i, sym in enumerate(pent2ene_syms) if sym == "C"]]
carbon_centroid = carbon_xyzs.mean(axis=0)
_, _, vh = np.linalg.svd(carbon_xyzs - carbon_centroid)
normal = vh[-1] / np.linalg.norm(vh[-1])

# Orient the normal to point toward H11's side of the backbone plane, so the
# hydroxyl fragment approaches from outside the molecule.
if np.dot(h11_xyz - carbon_centroid, normal) < 0:
    normal = -normal

o16_xyz = h11_xyz + O_H11_DISTANCE * normal
oh_bond_length = np.linalg.norm(hydroxyl_xyzs[1] - hydroxyl_xyzs[0])
h17_xyz = o16_xyz + oh_bond_length * normal
hydroxyl_xyzs = np.vstack([o16_xyz, h17_xyz])

complex_xyzs = np.vstack([pent2ene_xyzs, hydroxyl_xyzs])
complex_syms = pent2ene_syms + hydroxyl_syms
complex_geo = GeometryRow(
    symbols=complex_syms, coordinates=complex_xyzs, charge=0, spin=1
)

# Constrained optimization on the complex
with db.session() as sess:
    geom_block = BlockGeom(constraints="{ B 10 15 1.15 C }")
    constr_input = CalculationInput(
        memory=mem_mib, ncores=args.ncores, geom_block=geom_block
    )
    logger.info("Beginning OPT calculation.")
    work_dir = SCAN_DIR / "constrained_opt"
    work_dir.mkdir(parents=True, exist_ok=True)
    structure = geo_to_struc(complex_geo)
    calc_row, _, output = run_calculation(
        structure,
        work_dir=work_dir,
        model=revdsd_model,
        calc_type=calc_type,
        calc_input=constr_input,
    )
