"""Utility variables and functions."""

import datetime
import json
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


def structure_to_geometry(struc: Structure) -> GeometryRow:
    """Convert an OPI Structure to an AutoStorage Geometry."""
    return GeometryRow.from_xyz_block(
        struc.to_xyz_block(), charge=struc.charge, spin=struc.multiplicity + 1
    )


class CalculationType(StrEnum):
    """Enum of ORCA calculation types."""

    GOAT = "goat"
    OPT = "opt"


class CalculationInput(BaseModel):
    """Container for Calculation input provenance."""

    memory: int
    ncores: int


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

    if model.basis is not None:
        calc.input.add_simple_keywords(model.method, model.basis, calc_type)
    else:
        calc.input.add_simple_keywords(model.method, calc_type)

    calc.input.memory = calc_input.memory
    calc.input.ncores = calc_input.ncores

    calc.write_input()
    calc.run()

    output = calc.get_output()
    output.parse()

    outfile = output.get_outfile()
    # Verify that ORCA terminated normally
    if not output.terminated_normally():
        msg = f"ORCA calculation failed, see output file: {outfile}"
        raise RuntimeError(msg)
    # Verify that SCF converged
    if not output.scf_converged():
        msg = f"ORCA SCF failed to converge, see output file: {outfile}"
        raise RuntimeError(msg)
    # Verify that geometry optimization converged
    if not output.scf_converged():
        msg = (
            f"ORCA geometry optimization failed to converge, see output file: {outfile}"
        )
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
