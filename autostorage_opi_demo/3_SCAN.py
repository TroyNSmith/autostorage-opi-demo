"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

import sys

import numpy as np
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    CalculationTrajectoryLink,
    Database,
    EnergyRow,
    GeometryRow,
    GeometryTrajectoryLink,
    Role,
    StageRow,
    StationaryPointRow,
    StepRow,
    TrajectoryRow,
)
from opi.input.blocks import BlockGeom

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import B3LYP, HYDROXYL, PENT2ENE, WB97X, XTB, CalcInput, CalcType

FP_ERR = 1e-8  # Floating-point error

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
SCAN_DIR = const.OUT_DIR / "3_SCAN"
SCAN_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Get the rotation matrix that rotates unit vector a onto unit vector b."""
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < FP_ERR:
        # a and b are (anti)parallel; any axis perpendicular to a works for a
        # 180-degree flip, and no rotation is needed for a parallel pair.
        if c > 0:
            return np.eye(3)
        axis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def form_complex(
    geo1: GeometryRow, geo2: GeometryRow, atom1: int, atom2: int, dist: float
) -> GeometryRow:
    """Set the distance from atom1 on geo1 to atom2 on geo2 to dist.

    geo2 is rotated and translated as a rigid body, so its internal
    coordinates (e.g. bond lengths) are unaffected and none of geo1's
    internal coordinates are touched.

    The approach direction is the normal to the best-fit plane of geo1's
    atoms, oriented toward atom1's side of that plane, which keeps geo2 from
    overlapping with the rest of geo1. geo2 is rotated about atom2 so that
    the vector from atom2 toward the rest of geo2 points along that same
    direction, i.e. away from geo1.
    """
    geo1_xyzs = geo1.coordinates
    geo2_xyzs = geo2.coordinates

    atom1_xyz = geo1_xyzs[atom1]

    centroid1 = geo1_xyzs.mean(axis=0)
    _, _, vh = np.linalg.svd(geo1_xyzs - centroid1)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    if np.dot(atom1_xyz - centroid1, normal) < 0:
        normal = -normal

    atom2_xyz = geo2_xyzs[atom2]
    rest_mask = np.arange(len(geo2_xyzs)) != atom2
    rest_centroid = geo2_xyzs[rest_mask].mean(axis=0)
    direction = rest_centroid - atom2_xyz
    direction_norm = np.linalg.norm(direction)
    if direction_norm > FP_ERR:
        rotation = _rotation_aligning(direction / direction_norm, normal)
        geo2_xyzs = (geo2_xyzs - atom2_xyz) @ rotation.T + atom2_xyz

    target_xyz = atom1_xyz + dist * normal
    geo2_xyzs = geo2_xyzs + (target_xyz - geo2_xyzs[atom2])

    complex_xyzs = np.vstack([geo1_xyzs, geo2_xyzs])
    complex_syms = geo1.symbols + geo2.symbols
    return GeometryRow(
        symbols=complex_syms,
        coordinates=complex_xyzs,
        charge=geo1.charge + geo2.charge,
        spin=geo1.spin + geo2.spin,
    )


with db.session() as sess:
    wb97x_model = query.get_or_create_model(sess, WB97X)
    sess.add(wb97x_model)
    sess.commit()

    pent2ene_geo = utils.struc_to_geo(PENT2ENE)
    pent2ene_calc_id = query.calculation_by_inchi(
        sess, model=wb97x_model, calc_type=CalcType.OPT, geo=pent2ene_geo
    )
    pent2ene_calc = utils.row_from_id(sess, CalculationRow, pent2ene_calc_id)
    pent2ene_wb97x = utils.get_output_geometry(pent2ene_calc)

    hydroxyl_geo = utils.struc_to_geo(HYDROXYL)
    hydroxyl_calc_id = query.calculation_by_inchi(
        sess, model=wb97x_model, calc_type=CalcType.OPT, geo=hydroxyl_geo
    )
    hydroxyl_calc = utils.row_from_id(sess, CalculationRow, hydroxyl_calc_id)
    hydroxyl_wb97x = utils.get_output_geometry(hydroxyl_calc)

    # Build the [OH].C(C)C=CC H-abstraction pre-reactive complex
    complex_geo = form_complex(pent2ene_wb97x, hydroxyl_wb97x, 10, 0, 1.25)

    xtb_model = query.get_or_create_model(sess, model=XTB)
    sess.add(xtb_model)
    # Add the complex to this session
    sess.add(complex_geo)
    sess.flush()

    calc_id = query.calculation_by_inchi(
        sess, model=xtb_model, calc_type=CalcType.OPT, geo=complex_geo
    )

    if calc_id is not None:
        logger.info(
            "Pre-existing scan found (id = %s).",
            calc_id,
        )
        sys.exit(0)

    logger.info("Beginning OPT calculation.")

    calc_input = CalcInput(
        memory=args.memory,
        ncores=args.ncores,
        geom_block=BlockGeom(scan="B 10 15 = 1.25, 0.96, 15"),
    )
    calc_row, _, output = utils.run_calculation(
        complex_geo,
        work_dir=SCAN_DIR / "scan",
        model=xtb_model,
        calc_type=CalcType.OPT,
        calc_input=calc_input,
    )
    results_props = output.results_properties
    if not results_props:
        raise ValueError

    # Link input Geometry to Calculation
    cg_link_in = CalculationGeometryLink(
        calculation=calc_row, geometry=complex_geo, role=Role.INPUT
    )
    rows = [calc_row, cg_link_in]

    strucs, props = utils.parse_trj(SCAN_DIR / "scan", file_pattern="*.allxyz")

    # Instantiate a trajectory for the output
    trj_row = TrajectoryRow()
    # Link Trajectory to Calculation
    ct_link = CalculationTrajectoryLink(
        calculation=calc_row, trajectory=trj_row, role=Role.OUTPUT
    )
    rows.extend([trj_row, ct_link])

    # Start a list of the start, TS, and end stages
    stg_row1 = StageRow(
        stationaries=[
            pent2ene_wb97x.stationary_points[0],
            hydroxyl_wb97x.stationary_points[0],
        ]
    )
    stages = [stg_row1]
    for i, (struc, prop) in enumerate(zip(strucs, props, strict=True)):
        ene = prop.energy_total
        if ene is None:
            msg = f"Energy not found for {struc}."
            raise ValueError(msg)

        geo_row = utils.struc_to_geo(struc)
        geo_row.spin = complex_geo.spin  # Make sure spin is properly recorded
        ene_row = EnergyRow(calculation=calc_row, geometry=geo_row, value=ene)

        # Link output Geometry to Trajectory
        gt_link = GeometryTrajectoryLink(
            trajectory=trj_row, geometry=geo_row, index=[i]
        )
        rows.extend([geo_row, ene_row, gt_link])

        # Track the middle and ending structures as pseudo stationary points
        # and mark the stationary points as stages
        middle = int(len(strucs) / 2)
        if i in [middle, len(strucs) - 1]:
            stp_row = StationaryPointRow(
                calculation=calc_row,
                geometry=geo_row,
                order=i == middle,
                is_pseudo=True,
            )
            stg_row = StageRow(
                stationaries=[stp_row],
                is_ts=i == middle,
            )
            rows.extend([stp_row, stg_row])
            stages.append(stg_row)

    # Build a step connecting the beginning and pseudo TS/end stages
    step_row = StepRow(
        stage1=stages[0],
        stage_ts=stages[1],
        stage2=stages[2],
    )
    rows.append(step_row)

    sess.add_all(rows)
    sess.flush()
    sess.commit()
    sess.close()
