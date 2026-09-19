"""Intrinsic Reaction Coordinate validation."""

import sys

from automol.ident import RDKIT_INCHI
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    CalculationTrajectoryLink,
    Database,
    EnergyRow,
    GeometryRow,
    HessianRow,
    IdentityRow,
    Role,
    StageRow,
    StationaryPointRow,
    StepRow,
    TrajectoryRow,
)
from autostorage.models import IdentityStationaryLink
from orca_parser import HessianTools
from sqlmodel import select

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, XTB, CalcInput, CalcType

ts_inchi = "InChI=1S/C5H11O/c1-3-4-5(2)7-6/h3-6H,1-2H3/b4-3+/t5-/m1/s1"

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
IRC_DIR = const.OUT_DIR / "5_IRC"
IRC_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)


with db.session() as sess:
    XTB = query.get_or_create_model(sess, model=XTB)
    sess.add(XTB)
    HF3C = query.get_or_create_model(sess, model=HF3C)
    sess.add(HF3C)

    stmt = (
        select(GeometryRow)
        .join(StationaryPointRow)
        .join(CalculationRow)
        .join(IdentityStationaryLink)  # Need to include the link
        .join(IdentityRow)
        .where(
            GeometryRow.id == StationaryPointRow.geometry_id,
            CalculationRow.model_id == HF3C.id,
            CalculationRow.calc_type == CalcType.OPT_TS,
            IdentityRow.algorithm == RDKIT_INCHI,
            IdentityRow.value == ts_inchi,
        )
    )
    # .one() will raise an Error if len(scan_trj) != 1
    optts_geo: GeometryRow = sess.execute(stmt).one()[0]

    # Query for existing IRC
    stmt = (
        select(CalculationRow)
        .join(CalculationGeometryLink)
        .where(
            CalculationRow.model_id == HF3C.id,
            CalculationRow.calc_type == CalcType.IRC,
            CalculationGeometryLink.geometry_id == optts_geo.id,
            CalculationGeometryLink.role == Role.INPUT,
        )
    )
    irc_calc = sess.execute(stmt).first()
    if irc_calc is not None:
        logger.info(
            "Pre-existing IRC found (id = %s). Skipping calculation.",
            irc_calc[0].id,
        )
        sys.exit(0)

    # 1. IRC
    logger.info("Beginning IRC.")
    optts_calc, _, optts_output = utils.run_calculation(
        struc=optts_geo,
        work_dir=IRC_DIR / "irc",
        model=HF3C,
        calc_type=CalcType.IRC,
        calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
    )
