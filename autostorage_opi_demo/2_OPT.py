"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

from pathlib import Path

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
from sqlalchemy import and_
from sqlmodel import select, col

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, HYDROXYL, XTB, CalcInput, CalcType

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
                onclause=col(CalculationGeometryLink.calculation_id)
                == col(CalculationRow.id),
            )
            .join(
                GeometryRow,
                col(GeometryRow.id) == col(CalculationGeometryLink.geometry_id),
            )
            .where(
                col(CalculationRow.model_id) == XTB.id,
                col(CalculationRow.calc_type) == CalcType.GOAT,
                col(CalculationGeometryLink.role) == Role.INPUT,
                col(GeometryRow.id) == geo.id,
            )
        )
        opt_calc = sess.scalars(stmt).first()
        if opt_calc is not None:
            logger.info(
                "Pre-existing OPT found (id = %s). Skipping calculation.",
                opt_calc.id,
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
    stmt = (
        select(CalculationRow)
        .join(
            target=StationaryPointRow,
            onclause=col(CalculationRow.id) == col(StationaryPointRow.calculation_id),
        )
        .join(
            target=IdentityStationaryLink,
            onclause=StationaryPointRow.id == IdentityStationaryLink.stationary_id,  # ty: ignore[invalid-argument-type]
        )
        .join(
            target=IdentityRow,
            onclause=IdentityStationaryLink.identity_id == IdentityRow.id,  # ty: ignore[invalid-argument-type]
        )
        .where(
            CalculationRow.model_id == XTB.id,
            CalculationRow.calc_type == CalcType.GOAT,
            col(StationaryPointRow.is_pseudo).is_(False),
            IdentityRow.algorithm == "rdkit inchi",  # ty: ignore[invalid-argument-type]
            IdentityRow.value == "InChI=1S/C5H10/c1-3-5-4-2/h3,5H,4H2,1-2H3/b5-3+",  # ty: ignore[invalid-argument-type]
        )
    )
    goat_calc: CalculationRow | None = sess.scalars(stmt).first()
    if not goat_calc:
        msg = "GOAT calculation not found in database."
        raise LookupError(msg)

    goat_geos: list[GeometryRow] = [
        cgl.geometry
        for cgl in goat_calc.geometry_links
        if cgl.role == Role.OUTPUT  # Filter by output geometries
    ]
    pent2ene_min: GeometryRow = min(
        [
            e
            for g in goat_geos
            for e in g.energies
            if e.calculation.model_id == HF3C.id  # Filter by HF-3c energies
        ],
        key=lambda e: e.value,
    ).geometry

    logger.info("Lowest energy conformer identified (id = %s).", pent2ene_min.id)
    sess.close()

# Optimize pent2ene
optimize(db, HF3C, pent2ene_min, OPT_DIR / "pent2ene")

# Optimize hydroxyl
hydroxyl_geo = utils.struc_to_geo(HYDROXYL)
optimize(db, HF3C, hydroxyl_geo, OPT_DIR / "hydroxyl")
