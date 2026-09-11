"""Query functions."""

from automol import Algorithm
from autostorage import GeometryRow, IdentityRow, ModelRow
from sqlalchemy.orm import Session as SASession
from sqlmodel import Session as SMSession
from sqlmodel import select


def get_or_create_model(
    sess: SASession | SMSession,
    program: str,
    method: str,
    basis: str | None,
    program_version: str,
) -> ModelRow:
    """Get an existing model or create a new one if it doesn't exist.

    Parameters:
        sess: The database session to query and potentially add to.
        program: The computational program (e.g., "orca").
        method: The computational method (e.g., "xtb", "b3lyp").
        basis: The basis set (e.g., "def2-TZVP"), or None if not applicable.
        program_version: The version of the program (e.g., "6.1.1").

    Returns:
        The existing or new ModelRow instance.
    """
    # Query for existing model with matching parameters
    stmt = select(ModelRow).where(
        ModelRow.program == program,
        ModelRow.method == method,
        ModelRow.basis == basis,
        ModelRow.program_version == program_version,
    )
    existing = sess.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return existing

    # Otherwise return new model
    model = ModelRow(
        program=program,
        method=method,
        basis=basis,
        program_version=program_version,
    )

    sess.add(model)
    sess.flush()  # Ensure model has an ID
    sess.commit()
    return model


def calculation(
    sess: SASession | SMSession,
    model: ModelRow,
    calc_type: str,
    geo: GeometryRow,
) -> int | None:
    """Check if a calculation already exists for the given model, calc_type, and geo.

    NOTE: Calculation must be associated with a stationary point matching the geo
    InChI.

    Parameters:
        sess: The database session to query.
        model: The computational model to check for.
        calc_type: The type of calculation (e.g., "goat").
        geo: The molecular geometry to check for (uses InChI for comparison).

    Returns: Calculation.id or None
    """
    # Create a temporary GeometryRow to get the InChI
    target_identity = IdentityRow.from_geometry(geo, algorithm=Algorithm.RDKIT_INCHI)
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
