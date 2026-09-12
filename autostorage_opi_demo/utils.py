"""Utility variables and functions."""

from opi.input.blocks import Block, BlockGeom

from opi.input.simple_keywords import DispersionCorrection, RelativisticCorrection

import datetime
import json
import logging
import re
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

from autostorage import CalculationRow, GeometryRow, ModelRow
from opi.core import Calculator
from opi.input.structures import Structure
from opi.output.core import Output
from pydantic import BaseModel

OUT_DIR = Path(__file__).parent.parent / "out"
DB_PATH = OUT_DIR / "demo.db"

ORCA_EXE = shutil.which("orca")


class _GreenInfoFormatter(logging.Formatter):
    """Formatter that renders INFO-level records in green ANSI text."""

    GREEN = "\033[32m"
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno == logging.INFO:
            return f"{self.GREEN}{message}{self.RESET}"
        return message


def get_logger(name: str) -> logging.Logger:
    """Get a module-level logger that prints INFO messages in green.

    Args:
        name: Name of the logger, typically the calling module's __name__.

    Returns:
        A configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            _GreenInfoFormatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def get_orca_version() -> str:
    """Get the ORCA program version from the system.

    Returns:
        The ORCA version string (e.g., "6.1.1").
    """
    if not ORCA_EXE:
        msg = "ORCA Executable could not be found."
        raise FileNotFoundError(msg)

    result = subprocess.run(  # noqa: S603
        [ORCA_EXE, "--version"], capture_output=True, text=True, check=False
    )
    output = result.stdout + result.stderr

    # Look for "Program Version X.Y.Z"
    match = re.search(r"Program Version\s+([\d.]+)", output)
    if match:
        return match.group(1)

    msg = "Could not determine ORCA version"
    raise RuntimeError(msg)


ORCA_VERSION = get_orca_version()


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


class CalculationType(StrEnum):
    """Enum of ORCA calculation types."""

    GOAT = "goat"
    OPT = "opt"


class CalculationInput(BaseModel):
    """Container for Calculation input provenance."""

    memory: int
    ncores: int
    geom_block: BlockGeom | None = None


class ModelKeywords(BaseModel):
    """Container for Model keywords."""

    corrections: list[str] | None = None


def run_calculation(
    structure: Structure,
    work_dir: str | Path,
    model: ModelRow,
    calc_type: str,
    calc_input: CalculationInput,
) -> tuple[CalculationRow, Calculator, Output]:
    """Run an ORCA calculation for the given structure and parameters.

    Args:
        structure: The molecular structure to run the calculation on
        work_dir: The directory to store calculation results
        model: The computational model to use for the calculation
        calc_type: The type of calculation being run (e.g., "goat")
        calc_input: Calculation input provenance.

    Returns:
        CalculationRow
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    calc = Calculator(basename=calc_type, working_dir=work_dir)
    calc.structure = structure

    calc.input.memory = calc_input.memory
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
