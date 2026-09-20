"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

from pathlib import Path

from automol.ident import RDKIT_INCHI
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    Database,
    EnergyRow,
    GeometryRow,
    GradientRow,
    IdentityRow,
    ModelRow,
    Role,
    StationaryPointRow,
)
from autostorage.models import IdentityStationaryLink
from sqlmodel import select

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, HYDROXYL, PENT2ENE, XTB, CalcInput, CalcType

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
OPT_DIR = const.OUT_DIR / "2_OPT"
OPT_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)


def optimize(
    db: Database, model: ModelRow, geo: GeometryRow, work_dir: str | Path
) -> GeometryRow:
    """Optimize a geometry at model."""
    with db.session() as sess:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Ensure model and geo are in the session
        sess.add_all([model, geo])
        sess.flush()

        stmt = (
            select(CalculationRow)
            .join(
                CalculationGeometryLink,
                CalculationRow.id == CalculationGeometryLink.calculation_id,  # ty: ignore[invalid-argument-type]
            )
            .join(GeometryRow, GeometryRow.id == CalculationGeometryLink.geometry_id)  # ty: ignore[invalid-argument-type]
            .join(IdentityStationaryLink)  # Need to include the link
            .join(IdentityRow)
            .where(
                CalculationRow.model_id == XTB.id,
                CalculationRow.calc_type == CalcType.GOAT,
                CalculationGeometryLink.role == Role.INPUT,
                GeometryRow.id == geo.id,
            )
        )
        opt_calc = sess.execute(stmt).first()
        if opt_calc is not None:
            logger.info(
                "Pre-existing OPT found (id = %s). Skipping calculation.",
                opt_calc[0].id,
            )
            return next(
                cgl.geometry
                for cgl in opt_calc.geometry_links
                if cgl.role == Role.OUTPUT
            )

        logger.info("Beginning OPT calculation for geo %s.", geo.id)
        opt_calc, _, opt_output = utils.run_calculation(
            struc=geo,
            work_dir=work_dir,
            model=model,
            calc_type=CalcType.OPT,
            calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
        )
        # Link input Geometry to Calculation
        cg_link_in = CalculationGeometryLink(
            calculation=opt_calc, geometry=geo, role=Role.INPUT
        )
        sess.add_all([opt_calc, cg_link_in])

        struc = opt_output.get_structure()
        grad = opt_output.get_gradient(index=-2)  # Last gradient calculated
        ene = opt_output.get_final_energy()

        if not struc or not grad or not ene:
            msg = "Optimization output did not parse expected results."
            raise ValueError(msg)

        opt_geo = utils.struc_to_geo(struc)
        opt_ene = EnergyRow(calculation=opt_calc, geometry=opt_geo, value=ene)
        opt_gra = GradientRow(calculation=opt_calc, geometry=opt_geo, value=grad)
        opt_stp = StationaryPointRow(calculation=opt_calc, geometry=opt_geo, order=0)
        cgl_out = CalculationGeometryLink(
            calculation=opt_calc, geometry=opt_geo, role=Role.OUTPUT
        )

        sess.add_all([opt_geo, opt_ene, opt_gra, opt_stp, cgl_out])
        sess.commit()
        sess.close()

        return opt_geo


with db.session() as sess:
    # Ensure all models are in the database
    XTB = query.get_or_create_model(sess, XTB)
    HF3C = query.get_or_create_model(sess, HF3C)
    sess.add_all([XTB, HF3C])
    sess.commit()

    # Query whether the goat calculation exists by checking if pent2ene's InChI is
    # tagged to a calculation with model=xtb_model and calc_type="goat". Then, fetch the
    # geometry with the lowest energy for further optimization
    pent2ene_geo = utils.struc_to_geo(PENT2ENE)
    goat_id = query.calculation_by_inchi(
        sess, model=XTB, calc_type=CalcType.GOAT, geo=pent2ene_geo
    )
    goat_row = utils.row_from_id(sess, CalculationRow, goat_id)
    sess.merge(goat_row)

    min_ene = min(goat_row.energies, key=lambda e: e.value)
    pent2ene_min = utils.row_from_id(sess, GeometryRow, min_ene.geometry_id)
    logger.info("Lowest energy conformer identified (id = %s)", min_ene.geometry_id)

    sess.close()

# Optimize pent2ene
optimize(db, HF3C, pent2ene_min, OPT_DIR / "pent2ene")

# Optimize hydroxyl
hydroxyl_geo = utils.struc_to_geo(HYDROXYL)
optimize(db, HF3C, hydroxyl_geo, OPT_DIR / "hydroxyl")
