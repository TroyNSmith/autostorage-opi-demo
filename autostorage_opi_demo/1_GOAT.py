"""Global Optimization of Pent-2-ene with the xTB model."""

import sys

from autostorage import (
    CalculationGeometryLink,
    Database,
    EnergyRow,
    Role,
    StationaryPointRow,
)

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import PENT2ENE, XTB, CalcInput, CalcType

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
GOAT_DIR = const.OUT_DIR / "1_GOAT"
GOAT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)

with db.session() as sess:
    # Query for existing xtb model or create a new one and ensure it's in the session
    xtb_model = query.get_or_create_model(sess, model=XTB)
    sess.add(xtb_model)

    # Query whether the calculation exists by checking if pent2ene's InChI is tagged
    # to a calculation with model=xtb_model and calc_type="goat"
    pent2ene_geo = utils.struc_to_geo(PENT2ENE)
    calc_id = query.calculation_by_inchi(
        sess, model=xtb_model, calc_type=CalcType.GOAT, geo=pent2ene_geo
    )

    if calc_id is not None:
        logger.info(
            "Pre-existing GOAT calculation found (id = %s). Skipping calculation.",
            calc_id,
        )
        sys.exit(0)

    logger.info("Beginning pent2ene GOAT calculation.")
    calc_input = CalcInput(memory=args.memory, ncores=args.ncores)
    calc_row, calc, _ = utils.run_calculation(
        PENT2ENE,
        work_dir=GOAT_DIR,
        model=xtb_model,
        calc_type=CalcType.GOAT,
        calc_input=calc_input,
    )
    # Link input Geometry to Calculation
    cg_link_in = CalculationGeometryLink(
        calculation=calc_row, geometry=pent2ene_geo, role=Role.INPUT
    )
    out_rows = [calc_row, pent2ene_geo, cg_link_in]

    strucs, props = utils.parse_trj(GOAT_DIR)
    for struc, prop in zip(strucs, props, strict=True):
        ene = prop.energy_total
        if ene is None:
            msg = f"Energy not found for {struc}."
            raise ValueError(msg)

        geo_row = utils.struc_to_geo(struc)
        ene_row = EnergyRow(calculation=calc_row, geometry=geo_row, value=ene)
        stp_row = StationaryPointRow(calculation=calc_row, geometry=geo_row, order=0)

        # Link output Geometry to Calculation
        cg_link_out = CalculationGeometryLink(
            calculation=calc_row, geometry=geo_row, role=Role.OUTPUT
        )
        out_rows.extend([geo_row, ene_row, stp_row, cg_link_out])

    sess.add_all(out_rows)
    sess.commit()
    sess.close()
