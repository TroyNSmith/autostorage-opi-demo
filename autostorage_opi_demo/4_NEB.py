"""Nudged Elastic Band search for complex TS."""

import sys

import numpy as np
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    CalculationTrajectoryLink,
    Database,
    EnergyRow,
    GeometryTrajectoryLink,
    Role,
    TrajectoryRow,
)
from opi.input.blocks import BlockGeom, BlockNeb

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

    scan_id = query.calculation_by_inchi(
        sess, model=XTB, calc_type=CalcType.OPT, inchi=COMPLEX_INCHI
    )
    scan_calc = utils.row_from_id(sess, CalculationRow, scan_id)
    scan_calc = sess.merge(scan_calc)
    scan_trj = next(tl.trajectory for tl in scan_calc.trajectory_links)
    scan_trj = sess.merge(scan_trj)

    geo1 = next(tgl.geometry for tgl in scan_trj.geometry_links if tgl.index == [0])
    geo_ts = next(tgl.geometry for tgl in scan_trj.geometry_links if tgl.index == [1])
    geo2 = next(tgl.geometry for tgl in scan_trj.geometry_links if tgl.index == [2])

    B3LYP = query.get_or_create_model(sess, HF3C)

    calc_id = query.calculation_by_inchi(
        sess, model=XTB, calc_type=CalcType.NEB_TS, inchi=COMPLEX_INCHI
    )

    if calc_id is not None:
        logger.info(
            "Pre-existing NEB found (id = %s).",
            calc_id,
        )
        sys.exit(0)

    logger.info("Beginning NEB.")

    neb_end_xyzfile = NEB_DIR / "prod.xyz"
    geo2.xyz_file(path=neb_end_xyzfile)
    neb_ts_xyzfile = NEB_DIR / "ts.xyz"
    geo_ts.xyz_file(path=neb_ts_xyzfile)

    calc_input = CalcInput(
        memory=args.memory,
        ncores=args.ncores,
        blocks=[
            BlockNeb(neb_end_xyzfile=neb_end_xyzfile, ts=neb_ts_xyzfile, preopt=True)
        ],
    )
    calc_row, _, output = utils.run_calculation(
        utils.geo_to_struc(geo1),
        work_dir=NEB_DIR,
        model=XTB,
        calc_type=CalcType.NEB_TS,
        calc_input=calc_input,
    )
