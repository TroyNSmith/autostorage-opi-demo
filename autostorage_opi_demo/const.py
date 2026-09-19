"""Custom objects for autostorage-opi demo."""

import re
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

from autostorage import ModelRow
from opi.input.blocks import BlockGeom, BlockNeb
from opi.input.structures import Structure
from pydantic import BaseModel

OUT_DIR = Path(__file__).parent.parent / "out"
DB_PATH = OUT_DIR / "demo.db"

ORCA_EXE = shutil.which("orca")

# Structures from SMILES
PENT2ENE = Structure.from_smiles("CC=CCC", charge=0, multiplicity=1)
HYDROXYL = Structure.from_smiles("[OH]", charge=0, multiplicity=2)
COMPLEX = Structure.from_smiles("C[CH]C=CC.O", charge=0, multiplicity=2)


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


class CalcType(StrEnum):
    """Enum of ORCA calculation types."""

    GOAT = "goat"
    OPT = "opt"
    NEB_TS = "tight-neb-ts"
    SCAN_TS = "ScanTS"
    ENERGY = "Energy"


class CalcInput(BaseModel):
    """Container for Calculation input provenance."""

    memory: int  # GB
    ncores: int
    blocks: list[BlockGeom | BlockNeb] | None = None


class ModelKeywords(BaseModel):
    """Container for Model keywords."""

    corrections: list[str] | None = None


XTB = ModelRow(
    program="ORCA",
    method="xtb",
    basis=None,
    program_version=ORCA_VERSION,
)
HF3C = ModelRow(
    program="ORCA",
    method="B3LYP",
    basis="SV(P)",
    program_version=ORCA_VERSION,
)
HF3C = ModelRow(
    program="ORCA",
    method="HF-3c",
    basis=None,
    program_version=ORCA_VERSION,
)
