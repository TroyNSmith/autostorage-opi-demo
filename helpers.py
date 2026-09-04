"""Helpers for the autostorage-opi-demo."""

import datetime
import re
import subprocess
from pathlib import Path

from automol.ident import Algorithm
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    EnergyRow,
    GeometryRow,
    IdentityRow,
    ModelRow,
    Role,
    StationaryPointRow,
)
from opi.core import Calculator
from opi.input.structures import Properties, Structure
from sqlalchemy.orm import Session as SASession
from sqlmodel import Session as SMSession
from sqlmodel import SQLModel, select

MAX_MEM_MIB = 8000


def get_or_create_model(
    sess: SASession | SMSession,
    program: str,
    method: str,
    basis: str | None,
    program_version: str,
) -> ModelRow:
    """Get an existing model or create a new one if it doesn't exist.

    Parameters
    ----------
    sess
        The database session to query and potentially add to.
    program
        The computational program (e.g., "orca").
    method
        The computational method (e.g., "xtb", "b3lyp").
    basis
        The basis set (e.g., "def2-TZVP"), or None if not applicable.
    program_version
        The version of the program (e.g., "6.1.1").

    Returns
    -------
        The existing or newly created ModelRow instance.
    """
    # Query for existing model with matching parameters
    stmt = select(ModelRow).where(
        ModelRow.program == program,
        ModelRow.method == method,
        ModelRow.basis == basis,
        ModelRow.program_version == program_version,
    )
    existing_model = sess.execute(stmt).scalar_one_or_none()

    if existing_model is not None:
        return existing_model

    # Create and add new model if it doesn't exist
    new_model = ModelRow(
        program=program,
        method=method,
        basis=basis,
        program_version=program_version,
    )
    sess.add(new_model)
    sess.flush()  # Ensure model has an ID
    sess.commit()
    return new_model


def calculation_exists(
    sess: SASession | SMSession,
    model: ModelRow,
    calc_type: str,
    structure: Structure,
) -> int | None:
    """Check if a calculation already exists for the given model, calc_type, and structure.

    Parameters
    ----------
    sess
        The database session to query.
    model
        The computational model to check for.
    calc_type
        The type of calculation (e.g., "goat").
    structure
        The molecular structure to check for (uses InChI for comparison).

    Returns
    -------
        Calculation.id or None
    """
    # Create a temporary GeometryRow to get the InChI
    temp_geo = GeometryRow.from_xyz_block(
        structure.to_xyz_block(),
        charge=structure.charge,
        spin=structure.multiplicity + 1,
    )
    target_identity = IdentityRow.from_geometry(
        temp_geo, algorithm=Algorithm.RDKIT_INCHI
    )
    target_inchi = target_identity.value

    with sess:
        identity_stmt = select(IdentityRow).where(
            IdentityRow.algorithm == Algorithm.RDKIT_INCHI,
            IdentityRow.value == target_inchi,
        )
        identities = sess.execute(identity_stmt).all()
        for (ident,) in identities:
            if not ident.stationary_points:
                continue

            for stp in ident.stationary_points:
                sess.merge(stp)
                if (
                    stp.calculation
                    and stp.calculation.model_id == model.id
                    and stp.calculation.calc_type == calc_type
                ):
                    return stp.calculation.id

    return None


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

    calc_row = CalculationRow(
        model_id=model.id,
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
        stp_row = StationaryPointRow(calculation=calc_row, geometry=geo_row, order=1)
        out_rows.extend([geo_row, cg_link, ene_row, stp_row])

    return out_rows
