"""Utility variables and functions."""

import tempfile

import argparse
import json
import logging
from pathlib import Path
from uuid import UUID

from autostorage import CalculationRow, GeometryRow, ModelRow, Role
from opi.core import Calculator
from opi.input.structures import Properties, Structure
from opi.output.core import Output
from sqlalchemy.orm import Session
from sqlmodel import SQLModel

from const import CalcInput


def get_parser() -> argparse.ArgumentParser:
    """Get a module-level parser."""
    parser = argparse.ArgumentParser(
        prog="AutoStorage Demonstration",
        description="Run ORCA GOAT on pent2ene and store results in AutoStorage database.",
    )

    parser.add_argument(
        "-m",
        "--memory",
        help="Available memory in GB.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-n",
        "--ncores",
        help="Available number of CPU cores.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print SQL actions to terminal.",
        action="store_true",
    )
    return parser


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def struc_to_geo(struc: Structure) -> GeometryRow:
    """Convert an OPI Structure to an AutoStorage Geometry."""
    return GeometryRow.from_xyz_block(
        struc.to_xyz_block(), charge=struc.charge, spin=struc.multiplicity - 1
    )


def geo_to_struc(geo: GeometryRow) -> Structure:
    """Convert an AutoStorage Geometry to an OPI Structure."""
    return Structure.from_xyz_block(
        geo.xyz_block(), charge=geo.charge, multiplicity=geo.spin + 1
    )


def run_calculation(
    struc: Structure | GeometryRow,
    work_dir: str | Path,
    model: ModelRow,
    calc_type: str,
    calc_input: CalcInput,
) -> tuple[CalculationRow, Calculator, Output]:
    """Run an ORCA calculation for the given structure and parameters.

    Args:
        struc: The molecular structure to run the calculation on
        work_dir: The directory to store calculation results
        model: The computational model to use for the calculation
        calc_type: The type of calculation being run (e.g., "goat")
        calc_input: Calculation input provenance.

    Returns:
        CalculationRow
    """
    if isinstance(struc, GeometryRow):
        struc = geo_to_struc(struc)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    calc = Calculator(basename=calc_type, working_dir=work_dir)
    calc.structure = struc

    # Convert memory to MiB and reduce to account for ORCA over-consumption
    calc.input.memory = int(calc_input.memory * 953.674 * 0.75)
    calc.input.ncores = calc_input.ncores
    calc.input.add_simple_keywords(model.method, calc_type)

    if model.basis:
        calc.input.add_simple_keywords(model.basis)

    if model.keywords:
        corrections = model.keywords.get("corrections", None)
        if corrections:
            calc.input.add_simple_keywords(*corrections)

    if calc_input.geom_block:
        calc.input.add_blocks(calc_input.geom_block)

    calc.write_input()
    calc.run()

    output = calc.get_output()
    output.parse()

    outfile = output.get_outfile()
    # Verify that ORCA terminated normally
    if not output.terminated_normally():
        msg = f"ORCA calculation failed, see output file: {outfile}"
        raise RuntimeError(msg)

    outjson = next(outfile.parent.glob("*.property.json"), None)
    if outjson is not None:
        keys_to_remove = ["Geometries"]
        with outjson.open("r") as f:
            outprov: dict = json.load(f)
        for key in keys_to_remove:
            outprov.pop(key, None)
    else:
        outprov = {}

    calc_row = CalculationRow(
        model_id=model.id,
        calc_type=calc_type,
        input_provenance=calc_input.model_dump(),
        output_provenance=outprov,
    )

    return calc_row, calc, output


def parse_trj(
    work_dir: str | Path, *, file_pattern: str = "*.finalensemble.xyz"
) -> tuple[list[Structure], list[Properties]]:
    """Parse trj_xyz file format from ORCA output."""
    work_dir = Path(work_dir)

    trjxyz = next(work_dir.glob(file_pattern), None)
    if trjxyz is None:
        msg = f"{file_pattern} could not be located in {work_dir}."
        raise FileNotFoundError(msg)

    trj = trjxyz.read_text()
    trj = trj.replace(">\n", "")
    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tf:
        tf.write(trj)
        tf.flush()
        return Structure.from_trj_xyz(tf.name), Properties.from_trj_xyz(
            tf.name, mode="goat"
        )


def row_from_id[T: SQLModel](
    sess: Session, entity: type[T], ident: int | UUID | None
) -> T:
    """Get a row from the database from its ID."""
    if ident is None:
        msg = "pent2ene GOAT calculation not found."
        raise KeyError(msg)
    row = sess.get(entity, ident)
    if row is None:
        msg = f"Error getting {entity.__class__} with {ident = }"
        raise LookupError(msg)
    return row


def get_output_geometry(calc: CalculationRow) -> GeometryRow:
    """Get a single output Geometry from the calculation."""
    cg_links = calc.geometry_links
    geo_out = [cg.geometry for cg in cg_links if cg.role == Role.OUTPUT]
    if len(geo_out) != 1:
        msg = f"{len(geo_out)} output geometries for {calc.id = }, expected 1."
        raise ValueError(msg)
    return geo_out[0]
