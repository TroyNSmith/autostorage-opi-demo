"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

from pathlib import Path

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

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HYDROXYL, PENT2ENE, WB97X, XTB, CalcInput, CalcType

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
OPT_DIR = const.OUT_DIR / "2_OPT"
OPT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)


def optimize(
    db: Database, model: ModelRow, geo_in: GeometryRow, work_dir: str | Path
) -> GeometryRow:
    """Optimize a geometry at model."""
    with db.session() as sess:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Assumes that model and geo_in are already assigned IDs in the database
        model = sess.merge(model)
        geo_in = sess.merge(geo_in)

        calc_id = query.calculation_by_inchi(
            sess, model=model, calc_type=CalcType.OPT, geo=geo_in
        )

        if calc_id is not None:
            logger.info(
                "Pre-existing OPT calculation found (id = %s). Skipping calculation.",
                calc_id,
            )
            calc_row = utils.row_from_id(sess, CalculationRow, calc_id)
            return utils.get_output_geometry(calc_row)

        logger.info("Beginning OPT calculation.")
        structure = utils.geo_to_struc(geo_in)
        calc_input = CalcInput(memory=args.memory, ncores=args.ncores)
        calc_row, _, output = utils.run_calculation(
            structure,
            work_dir=work_dir,
            model=model,
            calc_type=CalcType.OPT,
            calc_input=calc_input,
        )
        # Link input Geometry to Calculation
        cg_link_in = CalculationGeometryLink(
            calculation=calc_row, geometry=geo_in, role=Role.INPUT
        )
        sess.add_all([calc_row, cg_link_in])

        struc = output.get_structure()
        grad = output.get_gradient(index=-2)  # Last gradient calculated
        ene = output.get_final_energy()

        if not struc or not grad or not ene:
            msg = "Optimization output did not return expected results."
            raise ValueError(msg)

        geo_out = utils.struc_to_geo(struc)
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


with db.session() as sess:
    # Ensure all models are in the database
    xtb = query.get_or_create_model(sess, XTB)
    wb97x = query.get_or_create_model(sess, WB97X)
    sess.add_all([xtb, wb97x])
    sess.commit()

    # Query whether the goat calculation exists by checking if pent2ene's InChI is
    # tagged to a calculation with model=xtb_model and calc_type="goat". Then, fetch the
    # geometry with the lowest energy for further optimization
    pent2ene_geo = utils.struc_to_geo(PENT2ENE)
    goat_id = query.calculation_by_inchi(
        sess, model=xtb, calc_type=CalcType.GOAT, geo=pent2ene_geo
    )
    goat_row = utils.row_from_id(sess, CalculationRow, goat_id)
    sess.merge(goat_row)

    min_ene = min(goat_row.energies, key=lambda e: e.value)
    pent2ene_min = utils.row_from_id(sess, GeometryRow, min_ene.geometry_id)
    logger.info("Lowest energy conformer identified (id = %s)", min_ene.geometry_id)

    sess.close()

# Optimize pent2ene
pent2ene_wb97 = optimize(db, wb97x, pent2ene_min, OPT_DIR / "pent2ene_wb97")

# Optimize hydroxyl
hydroxyl_geo = utils.struc_to_geo(HYDROXYL)
hydroxyl_wb97 = optimize(db, wb97x, hydroxyl_geo, OPT_DIR / "hydroxyl_wb97")
