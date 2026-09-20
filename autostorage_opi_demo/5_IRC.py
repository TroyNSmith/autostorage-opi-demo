"""Intrinsic Reaction Coordinate validation."""

import sys
from typing import Any

from automol.ident import RDKIT_INCHI
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    CalculationTrajectoryLink,
    Database,
    EnergyRow,
    GeometryRow,
    GeometryTrajectoryLink,
    GradientRow,
    HessianRow,
    IdentityRow,
    Role,
    StageRow,
    StationaryPointRow,
    StepRow,
    TrajectoryRow,
    ValidationRow,
)
from autostorage.models import IdentityStationaryLink, StageStationaryLink
from opi.input.structures import Properties, Structure
from orca_parser import HessianTools
from sqlalchemy import or_
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

while True:
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
            irc_calc: CalculationRow = irc_calc[0]
            logger.info(
                "Pre-existing IRC found (id = %s). Skipping calculation.",
                irc_calc.id,
            )
            sess.close()
            break

        # 1. IRC
        logger.info("Beginning IRC.")
        irc_calc, _, irc_output = utils.run_calculation(
            struc=optts_geo,
            work_dir=IRC_DIR / "irc",
            model=HF3C,
            calc_type=CalcType.IRC,
            calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
        )
        cg_link = CalculationGeometryLink(
            calculation=irc_calc, geometry=optts_geo, role=Role.INPUT
        )
        rows = [irc_calc, cg_link]

        # Parse the backwards trajectory
        trj_row_b = TrajectoryRow()
        ct_link = CalculationTrajectoryLink(
            calculation=irc_calc, trajectory=trj_row_b, role=Role.OUTPUT
        )
        rows.extend([trj_row_b, ct_link])
        strucs_b = Structure.from_trj_xyz(IRC_DIR / "irc/IRC_IRC_B_trj.xyz")
        props_b = Properties.from_trj_xyz(IRC_DIR / "irc/IRC_IRC_B_trj.xyz")
        for i, (struc, prop) in enumerate(zip(strucs_b, props_b, strict=True)):
            if prop.energy_total is None:
                msg = f"Energy not determined for IRC_B {i}."
                raise ValueError(msg)
            geo_row = utils.struc_to_geo(struc)
            geo_row.spin = optts_geo.spin
            ct_link = GeometryTrajectoryLink(
                geometry=geo_row, trajectory=trj_row_b, index=[i]
            )
            ene_row = EnergyRow(
                calculation=irc_calc, geometry=geo_row, value=prop.energy_total
            )
            rows.extend([geo_row, ct_link, ene_row])

        # Parse the forwards trajectory
        trj_row_f = TrajectoryRow()
        ct_link = CalculationTrajectoryLink(
            calculation=irc_calc, trajectory=trj_row_f, role=Role.OUTPUT
        )
        rows.extend([trj_row_f, ct_link])
        strucs_f = Structure.from_trj_xyz(IRC_DIR / "irc/IRC_IRC_F_trj.xyz")
        props_f = Properties.from_trj_xyz(IRC_DIR / "irc/IRC_IRC_F_trj.xyz")
        for i, (struc, prop) in enumerate(zip(strucs_f, props_f, strict=True)):
            if prop.energy_total is None:
                msg = f"Energy not determined for IRC_F {i}."
                raise ValueError(msg)
            geo_row = utils.struc_to_geo(struc)
            geo_row.spin = optts_geo.spin
            ct_link = GeometryTrajectoryLink(
                geometry=geo_row, trajectory=trj_row_f, index=[i]
            )
            ene_row = EnergyRow(
                calculation=irc_calc, geometry=geo_row, value=prop.energy_total
            )
            rows.extend([geo_row, ct_link, ene_row])

        # Add a ValidationRow to attach to the Step
        stmt = (
            select(StepRow)
            .join(StageRow, StepRow.stage_id_ts == StageRow.id)  # ty: ignore[invalid-argument-type]
            .join(StageStationaryLink, StageRow.id == StageStationaryLink.stage_id)  # ty: ignore[invalid-argument-type]
            .join(StationaryPointRow)
            .join(CalculationRow)
            .where(
                StageRow.is_ts == True,  # noqa: E712
                StationaryPointRow.geometry_id == optts_geo.id,
                CalculationRow.model_id == HF3C.id,
            )
        )
        step_row = sess.execute(stmt).first()
        if step_row is None:
            msg = "Could not identify Reaction step from 4_NEB."
            raise LookupError(msg)

        vld_row = ValidationRow(
            calculation=irc_calc, step=step_row[0], method="Full IRC"
        )
        rows.append(vld_row)

        sess.add_all(rows)
        sess.commit()
        sess.close()
        break

# 2. Optimize backwards minima
with db.session() as sess:
    HF3C = query.get_or_create_model(sess, model=HF3C)
    sess.add(HF3C)

    # Merge relevant rows from IRC
    irc_calc = sess.merge(irc_calc)
    step_row = next(val.step for val in irc_calc.validations)
    # Get conformer IDs from current stationary points in step
    conf_ids = {
        ident.value: stp.id
        for stg in [step_row.stage1, step_row.stage2]
        for stp in stg.stationaries
        for ident in stp.identities
        if ident.algorithm == "irmsd_conformer"
    }
    to_reconcile = {}
    rows = []
    for i, trj in enumerate([ctl.trajectory for ctl in irc_calc.trajectory_links]):
        geos = [gtl.geometry for gtl in trj.geometry_links]
        min_ene = min(
            [e for g in geos for e in g.energies if e.calculation.model_id == HF3C.id],
            key=lambda e: e.value,
        )
        min_geo = min_ene.geometry

        # Query for existing calculation
        stmt = (
            select(CalculationRow)
            .join(
                CalculationGeometryLink,
                CalculationRow.id == CalculationGeometryLink.calculation_id,  # ty: ignore[invalid-argument-type]
            )
            .join(
                GeometryRow,
                GeometryRow.id == CalculationGeometryLink.geometry_id,  # ty: ignore[invalid-argument-type]
            )
            .where(
                CalculationRow.model_id == HF3C.id,
                CalculationRow.calc_type == CalcType.OPT,
                GeometryRow.id == min_geo.id,
            )
        )
        opt_calc = sess.execute(stmt).first()
        if opt_calc is not None:
            opt_calc: CalculationRow = opt_calc[0]
            logger.info(
                "Pre-existing OPT found (id = %s). Skipping calculation.",
                opt_calc.id,
            )
            continue

        logger.info("Beginning OPT calculation.")
        opt_calc, _, opt_output = utils.run_calculation(
            struc=min_geo,
            work_dir=IRC_DIR / f"opt_{i}",
            model=HF3C,
            calc_type=CalcType.OPT,
            calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
        )
        # Link input Geometry to Calculation
        cg_link_in = CalculationGeometryLink(
            calculation=opt_calc, geometry=min_geo, role=Role.INPUT
        )
        sess.add_all([opt_calc, cg_link_in])

        struc = opt_output.get_structure()
        grad = opt_output.get_gradient(index=-2)  # Last gradient calculated
        ene = opt_output.get_final_energy()

        if not struc or not grad or not ene:
            msg = "Optimization output did not return expected results."
            raise ValueError(msg)

        opt_geo = utils.struc_to_geo(struc)
        opt_ene = EnergyRow(calculation=opt_calc, geometry=opt_geo, value=ene)
        opt_gra = GradientRow(calculation=opt_calc, geometry=opt_geo, value=grad)
        opt_stp = StationaryPointRow(calculation=opt_calc, geometry=opt_geo, order=0)

        cg_link_out = CalculationGeometryLink(
            calculation=opt_calc, geometry=opt_geo, role=Role.OUTPUT
        )

        sess.add_all([opt_geo, opt_ene, opt_gra, opt_stp, cg_link_out])
        sess.flush()  # Flush to generate the identities

        # Check if the IRC end points match the original guesses
        conf_id = next(
            i.value for i in opt_stp.identities if i.algorithm == "irmsd_conformer"
        )
        if conf_id in conf_ids:
            stp_id_old = conf_ids.pop(conf_id)
            stmt = (
                select(StationaryPointRow)
                .join(CalculationRow, StationaryPointRow.id == CalculationRow.id)  # ty: ignore[invalid-argument-type]
                .where(
                    StationaryPointRow.geometry_id == stp_id_old,
                    CalculationRow.model_id == HF3C.id,
                )
            )
            orig_stp: StationaryPointRow = sess.execute(stmt).one()[0]
            orig_stp.is_pseudo = False
            sess.flush([orig_stp])

            logger.info(
                "Original stationary point %s was validated and kept.", stp_id_old
            )

        else:
            to_reconcile[conf_id] = opt_stp.id

    # Replace invalid stationary points
    for (_, stp_id_new), (_, stp_id_old) in zip(
        to_reconcile.items(), conf_ids.items(), strict=True
    ):
        stmt = (
            select(StageStationaryLink)
            .join(
                StepRow,
                or_(
                    StepRow.stage_id1 == StageStationaryLink.stage_id,  # ty: ignore[invalid-argument-type]
                    StepRow.stage_id2 == StageStationaryLink.stage_id,  # ty: ignore[invalid-argument-type]
                ),
            )
            .where(
                StageStationaryLink.stationary_id == stp_id_old,
                StepRow.id == step_row.id,
            )
        )
        ss_link: StageStationaryLink = sess.execute(stmt).one()[0]
        # Swap the new stationary id for the old one in the stage link
        ss_link.stationary_id = stp_id_new
        sess.flush([ss_link])

        logger.info(
            "Replaced stationary point %s with %s for stage %s.",
            stp_id_old,
            stp_id_new,
            ss_link.stage_id,
        )

    # Refresh step_row with the new relationships
    sess.refresh(step_row)
    # 3. Vibrational analysis
    for i, geo in enumerate(
        [
            stp.geometry
            for stg in [step_row.stage1, step_row.stage2]
            for stp in stg.stationaries
        ]
    ):
        logger.info("Beginning FREQ %s.", i)
        freq_calc, _, freq_output = utils.run_calculation(
            struc=geo,
            work_dir=IRC_DIR / f"freq_{i}",
            model=HF3C,
            calc_type=CalcType.FREQ,
            calc_input=CalcInput(memory=args.memory, ncores=args.ncores),
        )
        cg_link_in = CalculationGeometryLink(
            calculation=freq_calc, geometry=geo, role=Role.INPUT
        )

        hess_tool = HessianTools(IRC_DIR / f"freq_{i}/Freq.hess")
        modes = hess_tool.normalmodes
        zpe = freq_output.get_zpe()
        if zpe is None:
            msg = "Zero point energy not determined."
            raise ValueError(msg)

        hess_row = HessianRow(calculation=freq_calc, geometry=geo, value=modes)
        zpe_row = EnergyRow(calculation=freq_calc, geometry=geo, value=zpe)
        rows.extend([freq_calc, cg_link_in, hess_row, zpe_row])

    sess.add_all(rows)
    sess.commit()
    sess.close()
