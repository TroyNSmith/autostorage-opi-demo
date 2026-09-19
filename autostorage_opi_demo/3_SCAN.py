"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

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
    StationaryPointRow,
    TrajectoryRow,
)
from opi.input.blocks import BlockGeom, Constraint, Constraints, Hybrid, NumList, TSMode

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, HF3C, HYDROXYL, PENT2ENE, XTB, CalcInput, CalcType

FP_ERR = 1e-8  # Floating-point error

COMPLEX_INCHI = "InChI=1S/C5H9.H2O/c1-3-5-4-2;/h3-5H,1-2H3;1H2"

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
SCAN_DIR = const.OUT_DIR / "3_SCAN"
SCAN_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)


with db.session() as sess:
    HF3C = query.get_or_create_model(sess, HF3C)
    sess.add(HF3C)

    pent2ene_geo = utils.struc_to_geo(PENT2ENE)
    pent2ene_calc_id = query.calculation_by_inchi(
        sess, model=HF3C, calc_type=CalcType.OPT, geo=pent2ene_geo
    )
    pent2ene_calc = utils.row_from_id(sess, CalculationRow, pent2ene_calc_id)
    pent2ene_wb97x = utils.get_output_geometry(pent2ene_calc)

    hydroxyl_geo = utils.struc_to_geo(HYDROXYL)
    hydroxyl_calc_id = query.calculation_by_inchi(
        sess, model=HF3C, calc_type=CalcType.OPT, geo=hydroxyl_geo
    )
    hydroxyl_calc = utils.row_from_id(sess, CalculationRow, hydroxyl_calc_id)
    hydroxyl_wb97x = utils.get_output_geometry(hydroxyl_calc)

    # Build the [OH].C(C)C=CC H-abstraction pre-reactive complex
    complex_geo = utils.form_complex(pent2ene_wb97x, hydroxyl_wb97x, 10, 0, 1.4)
    # Add the complex to this session
    sess.add(complex_geo)
    sess.flush()

    XTB = query.get_or_create_model(sess, model=XTB)
    sess.add(XTB)

    calc_id = query.calculation_by_inchi(
        sess, model=XTB, calc_type=CalcType.SCAN_TS, inchi=COMPLEX_INCHI
    )

    if calc_id is not None:
        logger.info(
            "Pre-existing scan found (id = %s). Skipping calculation.",
            calc_id,
        )
        sys.exit(0)

    logger.info("Beginning constrained OPT.")

    calc_input = CalcInput(
        memory=args.memory,
        ncores=args.ncores,
        blocks=[
            BlockGeom(
                constraints=Constraints(
                    constraints=[
                        Constraint(mode="C", atom1=3),
                        Constraint(mode="C", atom1=10),
                        Constraint(mode="C", atom1=15),
                    ]
                ),
            ),
        ],
    )
    const_calc, _, const_output = utils.run_calculation(
        complex_geo,
        work_dir=SCAN_DIR / "constrained_opt",
        model=XTB,
        calc_type=CalcType.OPT,
        calc_input=calc_input,
    )

    const_struc = const_output.get_structure()
    if const_struc is None:
        msg = "Structure could not be determined from constrained OPT."
        raise ValueError

    const_geo = utils.struc_to_geo(const_struc)
    const_geo.spin = complex_geo.spin  # Make sure spin is properly recorded

    cg_link_in = CalculationGeometryLink(
        calculation=const_calc, geometry=complex_geo, role=Role.INPUT
    )
    cg_link_out = CalculationGeometryLink(
        calculation=const_calc, geometry=const_geo, role=Role.OUTPUT
    )
    stp_row = StationaryPointRow(
        calculation=const_calc, geometry=const_geo, order=0, is_pseudo=True
    )
    rows = [const_calc, complex_geo, const_geo, cg_link_in, cg_link_out, stp_row]

    # SCAN

    logger.info("Beginning SCAN.")

    scan_input = CalcInput(
        memory=args.memory,
        ncores=args.ncores,
        blocks=[BlockGeom(scan="B 15 10 = 1.4, 1.0, 10")],
    )
    scan_calc, _, scan_output = utils.run_calculation(
        struc=const_geo,
        work_dir=SCAN_DIR / "scan",
        model=XTB,
        calc_type=CalcType.OPT,
        calc_input=scan_input,
    )

    cg_link_in = CalculationGeometryLink(
        calculation=scan_calc, geometry=complex_geo, role=Role.INPUT
    )
    # Instantiate a trajectory for the output
    trj_row = TrajectoryRow()
    # Link Trajectory to Calculation
    ct_link = CalculationTrajectoryLink(
        calculation=scan_calc, trajectory=trj_row, role=Role.OUTPUT
    )
    rows.extend([cg_link_in, trj_row, ct_link])

    strucs, _ = utils.parse_trj(SCAN_DIR / "scan", file_pattern="*.allxyz")
    for i, const_struc in enumerate(strucs):
        logger.info("Beginning ENERGY %s", i)
        geo_row = utils.struc_to_geo(const_struc)
        geo_row.spin = complex_geo.spin  # Make sure spin is properly recorded

        gt_link = GeometryTrajectoryLink(
            trajectory=trj_row, geometry=geo_row, index=[i]
        )
        rows.extend([geo_row, gt_link])

        # Compute the energies separately for a better profile vs. xTB
        ene_calc, _, ene_output = utils.run_calculation(
            geo_row,
            work_dir=SCAN_DIR / f"ene_{i}",
            model=HF3C,
            calc_type=CalcType.ENERGY,
            calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
        )
        cg_link_in = CalculationGeometryLink(
            calculation=ene_calc, geometry=geo_row, role=Role.INPUT
        )
        ene = ene_output.get_final_energy()
        if ene is None:
            msg = f"Energy not determined for scan point {i}"
            raise ValueError(msg)
        ene_row = EnergyRow(calculation=ene_calc, geometry=geo_row, value=ene)
        rows.extend([cg_link_in, ene_row])

    # Save the last point as a pseudo stationary point
    stp_row = StationaryPointRow(
        calculation=scan_calc, geometry=geo_row, order=0, is_pseudo=True
    )
    rows.append(stp_row)
    sess.flush()
    raise NotImplementedError(stp_row)
    sess.add_all(rows)
    sess.commit()
    sess.close()
