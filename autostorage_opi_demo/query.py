"""Query functions."""

from automol import rdkit_inchi
from autostorage import GeometryRow, IdentityRow, ModelRow
from sqlalchemy.orm import Session as SASession
from sqlmodel import Session as SMSession
from sqlmodel import select


def get_or_create_model(sess: SASession | SMSession, model: ModelRow) -> ModelRow:
    """Get an existing model or create a new one if it doesn't exist.

    Parameters:
        sess: The database session to query and potentially add to.
        program: The computational program (e.g., "orca").
        method: The computational method (e.g., "xtb", "b3lyp").
        basis: The basis set (e.g., "def2-TZVP"), or None if not applicable.
        program_version: The version of the program (e.g., "6.1.1").
        keywords: ModelRow keywords (e.g., auxiliary basis sets, SCF convergence, ...)

    Returns:
        The existing or new ModelRow instance.
    """
    # Query for existing model with matching parameters
    stmt = select(ModelRow).where(
        ModelRow.program == model.program,
        ModelRow.method == model.method,
        ModelRow.basis == model.basis,
        ModelRow.program_version == model.program_version,
        ModelRow.keywords == model.keywords,
    )

    existing = sess.scalars(stmt).one_or_none()
    if existing is not None:
        return existing

    sess.add(model)
    sess.flush()  # Ensure model has an ID
    return model


def calculation_by_inchi(
    sess: SASession | SMSession,
    model: ModelRow,
    calc_type: str,
    geo: GeometryRow | None = None,
    inchi: str | None = None,
) -> int | None:
    """Check if a calculation already exists for the given model, calc_type, and geo.

    NOTE: Calculation must be associated with a stationary point matching the geo
    InChI.

    Parameters:
        sess: The database session to query.
        model: The computational model to check for.
        calc_type: The type of calculation (e.g., "goat").
        geo: The molecular geometry to check for (uses InChI for comparison).
        inchi: InChI string to query.

    Returns: Calculation.id or None
    """
    if not (inchi or geo):
        msg = "InChI or GeometryRow required for query."
        raise ValueError(msg)

    if geo and not inchi:
        # Create a temporary GeometryRow to get the InChI
        inchi = rdkit_inchi.identity_fn(geo)

    identity_stmt = select(IdentityRow).where(
        IdentityRow.algorithm == rdkit_inchi.name,
        IdentityRow.value == inchi,
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
