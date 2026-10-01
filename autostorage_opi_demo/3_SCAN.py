"""Optimization of hydroxyl radical and lowest-energy pent2ene conformer."""

from sqlmodel.sql.expression import SelectOfScalar

import sys

from automol import hill_formula
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    CalculationTrajectoryLink,
    Database,
    GeometryRow,
    GeometryTrajectoryLink,
    IdentityAlgorithmRow,
    IdentityRow,
    IdentityStationaryLink,
    ModelRow,
    PropertyValueRow,
    Role,
    StationaryPointRow,
    TrajectoryRow,
    energy_property_kind,
)
from opi.input.blocks import BlockGeom, Constraint, Constraints
from sqlmodel import col, select

import const
import ident  # noqa: F401 Ensures the custom identity is being added to registry
import query
import utils
from const import HF3C, XTB, CalcInput, CalcType

parser = utils.get_parser()
args = parser.parse_args()
logger = utils.get_logger(__name__)

# Build the working directory
SCAN_DIR = const.OUT_DIR / "3_SCAN"
SCAN_DIR.mkdir(exist_ok=True, parents=True)

# Initialize the database
db = Database(const.OUT_DIR / "demo.db", echo=args.verbose)

# InChIs for lookup
pent2ene_inchi = "InChI=1S/C5H10/c1-3-5-4-2/h3,5H,4H2,1-2H3/b5-3+"
hydroxyl_inchi = "InChI=1S/HO/h1H"
complex_inchi = "InChI=1S/C5H9.H2O/c1-3-5-4-2;/h3-5H,1-2H3;1H2"


with db.session() as sess:
    XTB = query.get_or_create_model(sess, model=XTB)
    HF3C = query.get_or_create_model(sess, HF3C)
    sess.add_all([XTB, HF3C])

    def optimized_geometry_by_inchi(model: ModelRow, inchi: str) -> GeometryRow:
        """Select a (non-pseudo) stationary point Geometry by InChI."""
        stmt = (
            select(GeometryRow)
            .join(
                target=StationaryPointRow,
                onclause=col(StationaryPointRow.geometry_id) == col(GeometryRow.id),
            )
            .join(
                target=CalculationRow,
                onclause=col(CalculationRow.id)
                == col(StationaryPointRow.calculation_id),
            )
            .join(
                target=IdentityStationaryLink,
                onclause=col(IdentityStationaryLink.stationary_id)
                == col(StationaryPointRow.id),
            )
            .join(
                target=IdentityRow,
                onclause=col(IdentityRow.id) == col(IdentityStationaryLink.identity_id),
            )
            .join(
                target=IdentityAlgorithmRow,
                onclause=col(IdentityAlgorithmRow.id) == col(IdentityRow.algorithm_id),
            )
            .where(
                col(CalculationRow.model_id) == model.id,
                col(CalculationRow.calc_type) == CalcType.OPT,
                col(StationaryPointRow.is_pseudo).is_(False),
                col(IdentityRow.value) == inchi,
                col(IdentityAlgorithmRow.name) == "rdkit inchi",
            )
        )
        return sess.scalars(stmt).one()

    pent2ene_opt = optimized_geometry_by_inchi(HF3C, pent2ene_inchi)
    hydroxyl_opt = optimized_geometry_by_inchi(HF3C, hydroxyl_inchi)

    # Build the [HO]~[H]C(C)C=CC H-abstraction pre-reactive complex
    complex_geo = utils.form_complex(pent2ene_opt, hydroxyl_opt, 10, 0, 1.4)
    # Add the complex to this session
    sess.add(complex_geo)
    sess.flush()

    stmt = (
        select(CalculationRow)
        .join(
            target=StationaryPointRow,
            onclause=col(StationaryPointRow.calculation_id) == col(CalculationRow.id),
        )
        .join(
            target=IdentityStationaryLink,
            onclause=col(IdentityStationaryLink.stationary_id)
            == col(StationaryPointRow.id),
        )
        .join(
            target=IdentityRow,
            onclause=col(IdentityRow.id) == col(IdentityStationaryLink.identity_id),
        )
        .join(
            target=IdentityAlgorithmRow,
            onclause=col(IdentityAlgorithmRow.id) == col(IdentityRow.algorithm_id),
        )
        .where(
            col(CalculationRow.model_id) == XTB.id,
            col(CalculationRow.calc_type) == CalcType.OPT,
            col(StationaryPointRow.is_pseudo).is_(False),
            col(IdentityRow.value) == "C5H11O",
            col(IdentityAlgorithmRow.name) == hill_formula.name,
        )
    )
    scan_calc: CalculationRow | None = sess.scalars(stmt).one_or_none()
    if scan_calc is not None:
        logger.info(
            "Pre-existing SCAN found (id = %s). Skipping calculation.",
            scan_calc.id,
        )
        sys.exit(0)

    logger.info("Beginning constrained OPT.")

    const_input = CalcInput(
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
        calc_input=const_input,
    )

    const_struc = const_output.get_structure()
    if const_struc is None:
        msg = "Structure could not be determined from constrained OPT output."
        raise ValueError

    # Make sure spin is properly recorded
    const_geo = utils.struc_to_geo(const_struc, spin=complex_geo.spin)

    cgl_in = CalculationGeometryLink(
        calculation=const_calc, geometry=complex_geo, role=Role.INPUT
    )
    cgl_out = CalculationGeometryLink(
        calculation=const_calc, geometry=const_geo, role=Role.OUTPUT
    )
    rows = [const_calc, complex_geo, const_geo, cgl_in, cgl_out]

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

    cgl_in = CalculationGeometryLink(
        calculation=scan_calc, geometry=complex_geo, role=Role.INPUT
    )
    # Instantiate a trajectory for the output
    trj_row = TrajectoryRow()
    # Link Trajectory to Calculation
    ctl_out = CalculationTrajectoryLink(
        calculation=scan_calc, trajectory=trj_row, role=Role.OUTPUT
    )
    rows.extend([cgl_in, trj_row, ctl_out])

    strucs, _ = utils.parse_trj(SCAN_DIR / "scan", file_pattern="*.allxyz")
    for i, const_struc in enumerate(strucs):
        logger.info("Beginning ENERGY %s", i)
        # Make sure spin is properly recorded
        geo_row = utils.struc_to_geo(const_struc, spin=complex_geo.spin)

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
        cgl_in = CalculationGeometryLink(
            calculation=ene_calc, geometry=geo_row, role=Role.INPUT
        )
        ene = ene_output.get_final_energy()
        if ene is None:
            msg = f"Energy not determined for scan point {i}"
            raise ValueError(msg)
        ene_row = PropertyValueRow(
            property_kind_name=energy_property_kind.name,
            calculation=ene_calc,
            geometry=geo_row,
            value=ene,
        )
        rows.extend([ene_calc, cgl_in, ene_row])

    sess.add_all(rows)
    sess.commit()
    sess.close()
