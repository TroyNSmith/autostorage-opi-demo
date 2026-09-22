"""Global Optimization of Pent2ene with the xTB model."""

from automol import rdkit_inchi

import sys

from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    Database,
    EnergyRow,
    IdentityRow,
    Role,
    StationaryPointRow,
    IdentityAlgorithmRow,
)
from autostorage.models import IdentityStationaryLink
from opi.input.structures import Structure
from sqlmodel import select, col, text

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, XTB, CalcInput, CalcType

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
GOAT_DIR = const.OUT_DIR / "1_GOAT"
GOAT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)

# Pent2ene InChI for lookup
pent2ene_inchi = "InChI=1S/C5H10/c1-3-5-4-2/h3,5H,4H2,1-2H3/b5-3+"

with db.session() as sess:
    # Query for existing xtb/hf-3c models or create new ones and add to session
    XTB = query.get_or_create_model(sess, model=XTB)
    sess.add(XTB)
    HF3C = query.get_or_create_model(sess, model=HF3C)
    sess.add(HF3C)

    # Query whether the calculation exists by checking if pent2ene's InChI is tagged
    # to a calculation with model_id==XTB.id and calc_type==CalcType.GOAT ("goat")
    stmt = (
        select(CalculationRow)
        .join(
            StationaryPointRow,
            onclause=col(StationaryPointRow.calculation_id) == col(CalculationRow.id),
        )  # Identity is linked to the Stationary
        .join(
            IdentityStationaryLink,
            onclause=col(IdentityStationaryLink.stationary_id)
            == col(StationaryPointRow.id),
        )  # Need to include the link
        .join(
            IdentityRow,
            onclause=col(IdentityRow.id) == col(IdentityStationaryLink.identity_id),
        )
        .join(
            IdentityAlgorithmRow,
            onclause=col(IdentityAlgorithmRow.id) == col(IdentityRow.algorithm_id),
        )
        .where(
            col(CalculationRow.model_id) == XTB.id,
            col(CalculationRow.calc_type) == CalcType.GOAT,
            col(IdentityRow.value) == pent2ene_inchi,
            col(IdentityAlgorithmRow.name) == rdkit_inchi.name,
        )
    )
    goat_calc = sess.scalars(stmt).first()
    if goat_calc is not None:
        logger.info(
            "Pre-existing GOAT calculation found (id = %s). Skipping calculation.",
            goat_calc.id,
        )
        sys.exit(0)

    # Initialize Structure and GeometryRow
    pent2ene_struc = Structure.from_smiles("CC=CCC")
    pent2ene_geo = utils.struc_to_geo(pent2ene_struc)

    logger.info("Beginning pent2ene GOAT calculation.")
    goat_calc, goat_calculator, _ = utils.run_calculation(
        pent2ene_geo,
        work_dir=GOAT_DIR / "goat",
        model=XTB,
        calc_type=CalcType.GOAT,
        calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
    )
    # Link input Geometry to Calculation
    cg_link_in = CalculationGeometryLink(
        calculation=goat_calc, geometry=pent2ene_geo, role=Role.INPUT
    )
    rows = [goat_calc, pent2ene_geo, cg_link_in]

    for i, struc in enumerate(
        Structure.from_trj_xyz(GOAT_DIR / "goat/goat.finalensemble.xyz")
    ):
        geo_row = utils.struc_to_geo(struc)
        # Store the Geometry as a Stationary Point
        stp_row = StationaryPointRow(calculation=goat_calc, geometry=geo_row, order=0)
        # Link output Geometry to Calculation
        cg_link_out = CalculationGeometryLink(
            calculation=goat_calc, geometry=geo_row, role=Role.OUTPUT
        )
        rows.extend([geo_row, stp_row, cg_link_out])

        # Compute the energies separately for a better profile vs. xTB
        ene_calc, _, ene_output = utils.run_calculation(
            geo_row,
            work_dir=GOAT_DIR / f"ene_{i}",
            model=HF3C,
            calc_type=CalcType.ENERGY,
            calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
        )
        cg_link_in = CalculationGeometryLink(
            calculation=ene_calc, geometry=geo_row, role=Role.INPUT
        )
        ene = ene_output.get_final_energy()
        if ene is None:
            msg = f"Energy not determined for conformer {i}."
            raise ValueError(msg)

        ene_row = EnergyRow(calculation=ene_calc, geometry=geo_row, value=ene)
        rows.extend([ene_calc, cg_link_in, ene_row])

    sess.add_all(rows)
    # Flush and enter (commit) the new rows
    # NOTE: It's safer and more efficient to commit once per session
    sess.commit()
    sess.close()
