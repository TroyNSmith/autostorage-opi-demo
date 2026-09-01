"""Helpers for the autostorage-opi-demo."""

import datetime
import re
import subprocess
from pathlib import Path

from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    EnergyRow,
    GeometryRow,
    ModelRow,
    Role,
)
from opi.core import Calculator
from opi.input.structures import Properties, Structure
from sqlmodel import SQLModel

MAX_MEM_MIB = 7600


def get_orca_version() -> str:
    """Get the ORCA program version from the system.

    Returns
    -------
        The ORCA version string (e.g., "6.1.1").
    """
    result = subprocess.run(
        ["orca", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr

    # Look for "Program Version X.Y.Z"
    match = re.search(r"Program Version\s+([\d.]+)", output)
    if match:
        return match.group(1)

    raise RuntimeError("Could not determine ORCA version")


def run_goat(
    structure: Structure, work_dir: str | Path, model: ModelRow
) -> list[SQLModel]:
    """Run the GOAT calculation for the given structure.

    Parameters
    ----------
        structure
            The molecular structure to run the calculation on.
        work_dir
            The directory to store calculation results.
        model
            The computational model to use for the calculation.

    Returns
    -------
        A list of SQLModel instances representing the calculation results.
    """
    calc = Calculator(basename="goat", working_dir=work_dir)
    calc.structure = structure

    if model.basis is not None:
        calc.input.add_simple_keywords(model.method, model.basis, "goat")
    else:
        calc.input.add_simple_keywords(model.method, "goat")

    calc.input.memory = MAX_MEM_MIB

    calc.write_input()
    calc.run()

    output = calc.get_output()
    status = output.terminated_normally()

    if not status:
        raise RuntimeError("GOAT calculation did not terminate normally.")

    # Add program version to model
    calc_row = CalculationRow(
        model=model,
        calc_type="goat",
        input_provenance={
            "max_mem": f"{MAX_MEM_MIB} MiB",
            "date_created": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        },
    )

    structures = Structure.from_trj_xyz(work_dir / f"{calc.basename}.finalensemble.xyz")
    properties = Properties.from_trj_xyz(
        work_dir / f"{calc.basename}.finalensemble.xyz", mode="goat"
    )

    out_rows = [calc_row]
    for struc, prop in zip(structures, properties):
        if prop.energy_total is None:
            raise ValueError("Energy total is None for a determined structure.")

        geo_row = GeometryRow.from_xyz_block(
            struc.to_xyz_block(), charge=struc.charge, spin=struc.multiplicity + 1
        )
        cg_link = CalculationGeometryLink(
            calculation=calc_row, geometry=geo_row, role=Role.OUTPUT
        )
        ene_row = EnergyRow(
            calculation=calc_row, geometry=geo_row, value=prop.energy_total
        )
        out_rows.extend([geo_row, cg_link, ene_row])

    return out_rows
