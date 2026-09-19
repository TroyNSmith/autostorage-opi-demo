"""Nudged Elastic Band search for complex TS."""

import sys

from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    CalculationTrajectoryLink,
    Database,
    EnergyRow,
    GeometryRow,
    HessianRow,
    Role,
    StageRow,
    StationaryPointRow,
    StepRow,
    TrajectoryRow,
)
from orca_parser import HessianTools
from sqlmodel import select

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, XTB, CalcInput, CalcType

FP_ERR = 1e-8  # Floating-point error

COMPLEX_INCHI = "InChI=1S/C5H9.H2O/c1-3-5-4-2;/h3-5H,1-2H3;1H2"

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
NEB_DIR = const.OUT_DIR / "4_NEB"
NEB_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)


with db.session() as sess:
    XTB = query.get_or_create_model(sess, model=XTB)
    sess.add(XTB)
    HF3C = query.get_or_create_model(sess, model=HF3C)
    sess.add(HF3C)

    stmt = (
        select(TrajectoryRow)
        .join(CalculationTrajectoryLink)
        .join(CalculationRow)
        .where(
            CalculationTrajectoryLink.role == Role.OUTPUT,
            CalculationRow.model_id == XTB.id,
            CalculationRow.calc_type == CalcType.OPT,
        )
    )
    # .one() will raise an Error if len(scan_trj) != 1
    scan_trj: TrajectoryRow = sess.execute(stmt).one()[0]
    scan_calc = scan_trj.calculation_links[0].calculation  # There should only be one
    geo_rows = [gtl.geometry for gtl in scan_trj.geometry_links]
    ene_rows = [
        e for g in geo_rows for e in g.energies if e.calculation.model_id == HF3C.id
    ]

    geo1: GeometryRow = min(scan_trj.geometry_links, key=lambda gtl: gtl.index).geometry
    geo2: GeometryRow = max(scan_trj.geometry_links, key=lambda gtl: gtl.index).geometry
    geo_ts: GeometryRow = max(ene_rows, key=lambda e: e.value).geometry

    if geo_ts.id in {geo1.id, geo2.id}:
        msg = "Highest point in energy scan is the first or last point."
        raise ValueError(msg)

    # Query for existing opt-ts
    stmt = (
        select(CalculationRow)
        .join(CalculationGeometryLink)
        .where(
            CalculationRow.calc_type == CalcType.OPT_TS,
            CalculationGeometryLink.geometry_id == geo_ts.id,
            CalculationGeometryLink.role == Role.INPUT,
        )
    )
    optts_calc = sess.execute(stmt).first()
    if optts_calc is not None:
        logger.info(
            "Pre-existing OPTTS found (id = %s). Skipping calculation.",
            optts_calc[0].id,
        )
        sys.exit(0)

    # 1. TS Optimization
    logger.info("Beginning OPTTS.")
    optts_calc, _, optts_output = utils.run_calculation(
        struc=geo_ts,
        work_dir=NEB_DIR / "optts",
        model=HF3C,
        calc_type=CalcType.OPT_TS,
        calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
    )
    optts_struc = optts_output.get_structure()
    if optts_struc is None:
        msg = "Structure could not be determined from constrained OPT."
        raise ValueError

    optts_geo = utils.struc_to_geo(optts_struc)
    optts_geo.spin = geo_ts.spin  # Make sure spin is properly recorded

    cg_link_in = CalculationGeometryLink(
        calculation=optts_calc, geometry=geo_ts, role=Role.INPUT
    )
    cg_link_out = CalculationGeometryLink(
        calculation=optts_calc, geometry=optts_geo, role=Role.OUTPUT
    )
    rows = [optts_calc, optts_geo, cg_link_in, cg_link_out]

    # Build an elementary step connecting geo1, geo2, and optts_geo
    stp_row1 = StationaryPointRow(
        calculation=scan_calc,
        geometry=geo1,
        order=0,
        is_pseudo=True,  # Pseudo because it's a scan point
    )
    stg_row1 = StageRow(stationaries=[stp_row1])

    stp_row2 = StationaryPointRow(
        calculation=scan_calc,
        geometry=geo2,
        order=0,
        is_pseudo=True,  # Pseudo because it's a scan point
    )
    stg_row2 = StageRow(stationaries=[stp_row2])

    stp_row_ts = StationaryPointRow(
        calculation=optts_calc,
        geometry=optts_geo,
        order=1,
        is_pseudo=False,  # Geometry is optimized into a maximum
    )
    stg_row_ts = StageRow(stationaries=[stp_row_ts], is_ts=True)

    step_row = StepRow(stage1=stg_row1, stage2=stg_row2, stage_ts=stg_row_ts)

    rows.extend(
        [stp_row1, stg_row1, stp_row2, stg_row2, stp_row_ts, stg_row_ts, step_row]
    )

    # 2. Vibrational analysis
    logger.info("Beginning FREQ.")
    freq_calc, _, freq_output = utils.run_calculation(
        struc=optts_geo,
        work_dir=NEB_DIR / "freq",
        model=HF3C,
        calc_type=CalcType.FREQ,
        calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
    )
    cg_link_in = CalculationGeometryLink(
        calculation=freq_calc, geometry=optts_geo, role=Role.INPUT
    )

    hess_tool = HessianTools(NEB_DIR / "freq/Freq.hess")
    modes = hess_tool.normalmodes
    zpe = freq_output.get_zpe()
    if zpe is None:
        msg = "Zero point energy not determined."
        raise ValueError(msg)

    hess_row = HessianRow(calculation=freq_calc, geometry=optts_geo, value=modes)
    zpe_row = EnergyRow(calculation=freq_calc, geometry=optts_geo, value=zpe)
    rows.extend([freq_calc, cg_link_in, hess_row, zpe_row])

    # 3. Single point energy
    logger.info("Beginning ENE.")
    ene_calc, _, ene_output = utils.run_calculation(
        struc=optts_geo,
        work_dir=NEB_DIR / "ene",
        model=HF3C,
        calc_type=CalcType.ENERGY,
        calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
    )
    cg_link_in = CalculationGeometryLink(
        calculation=ene_calc, geometry=optts_geo, role=Role.INPUT
    )
    ene = ene_output.get_final_energy()
    if ene is None:
        msg = "Single point energy not determined."
        raise ValueError(msg)

    ene_row = EnergyRow(calculation=ene_calc, geometry=optts_geo, value=ene)
    rows.extend([cg_link_in, ene_row])

    sess.add_all(rows)
    sess.commit()
    sess.close()
