"""Shared setup and autostorage recipes for the demo."""

import argparse
from pathlib import Path

import numpy as np
from automol import rdkit_smiles
from autostorage import (
    AutostorageSession,
    CalculationGeometryLink,
    GeometryRow,
    ModelRow,
    PropertyValueRow,
    Role,
    StationaryPointRow,
)
from sqlmodel import select

import extensions  # noqa: F401  Registers the custom identity and property kind
import orca

OUT_DIR = Path(__file__).parent.parent / "out"
DB_PATH = OUT_DIR / "demo.db"

PENT2ENE_SMILES = "C/C=C/CC"
PENT2ENE_INCHI = "InChI=1S/C5H10/c1-3-5-4-2/h3,5H,4H2,1-2H3/b5-3+"
HYDROXYL_SMILES = "[OH]"
HYDROXYL_INCHI = "InChI=1S/HO/h1H"
COMPLEX_FORMULA = "C5H11O"  # Hill formula of the pent-2-ene + OH complex

_parser = argparse.ArgumentParser(description="Run the autostorage demo.")
_parser.add_argument("-m", "--memory", type=int, default=8, help="Memory in GB.")
_parser.add_argument("-n", "--ncores", type=int, default=1, help="CPU cores.")
_parser.add_argument("-v", "--verbose", action="store_true", help="Echo SQL.")
# Ignore unknown arguments, such as those a Jupyter kernel is launched with
ARGS, _ = _parser.parse_known_args()
orca.configure(memory=ARGS.memory, ncores=ARGS.ncores)


def get_or_create_model(sess: AutostorageSession, method: str) -> ModelRow:
    """Get the ORCA model for `method`, adding it to the session if new."""
    stmt = select(ModelRow).where(
        ModelRow.program == "ORCA",
        ModelRow.program_version == orca.version(),
        ModelRow.method == method,
    )
    model = sess.exec(stmt).first()
    if model is None:
        model = ModelRow(program="ORCA", program_version=orca.version(), method=method)
        sess.add(model)
    return model


def from_smiles(smiles: str) -> GeometryRow:
    """Embed a geometry from a SMILES string."""
    return GeometryRow(**rdkit_smiles.geometry_fn(smiles).model_dump())


def energy(geo: GeometryRow, model: ModelRow) -> float:
    """Return the energy stored for `geo` at `model`."""
    return next(
        float(prop.value)
        for prop in geo.properties
        if prop.property_kind_name == "energy" and prop.calculation.model == model
    )


def optimize(
    sess: AutostorageSession, geo: GeometryRow, model: ModelRow, work_dir: Path
) -> StationaryPointRow:
    """Optimize `geo` and store the result as a minimum with its energy and gradient.

    Identities (InChI, SMILES, ...) are attached to the new stationary point
    automatically when the session is flushed.
    """
    opt = orca.run(geo, model, "opt", work_dir)
    opt_geo = opt.geometry()
    opt.calculation.geometry_links.append(
        CalculationGeometryLink(geometry=opt_geo, role=Role.OUTPUT)
    )
    minimum = StationaryPointRow(calculation=opt.calculation, geometry=opt_geo)
    sess.add_all(
        [
            minimum,
            PropertyValueRow(
                property_kind_name="energy",
                calculation=opt.calculation,
                geometry=opt_geo,
                value=opt.energy(),
            ),
            PropertyValueRow(
                property_kind_name="gradient",
                calculation=opt.calculation,
                geometry=opt_geo,
                value=opt.gradient(),
            ),
        ]
    )
    return minimum


def validate(
    sess: AutostorageSession, stp: StationaryPointRow, model: ModelRow, work_dir: Path
) -> None:
    """Compute frequencies for a stationary point and mark it as validated.

    autostorage refuses to flush a validated stationary point without a Hessian.
    """
    freq = orca.run(stp.geometry, model, "Freq", work_dir)
    sess.add_all(
        [
            PropertyValueRow(
                property_kind_name="hessian",
                calculation=freq.calculation,
                geometry=stp.geometry,
                value=freq.hessian(),
            ),
            PropertyValueRow(
                property_kind_name="zpe",
                calculation=freq.calculation,
                geometry=stp.geometry,
                value=freq.zpe(),
            ),
        ]
    )
    stp.is_validated = True


def form_complex(
    geo1: GeometryRow, geo2: GeometryRow, atom1: int, atom2: int, dist: float
) -> GeometryRow:
    """Place `geo2` so that its `atom2` is `dist` from `geo1`'s `atom1`.

    `geo2` is moved rigidly along the normal to `geo1`'s best-fit plane (on
    `atom1`'s side), and rotated so the rest of `geo2` points away from `geo1`.
    """
    xyz1, xyz2 = geo1.coordinates, geo2.coordinates
    centroid = xyz1.mean(axis=0)
    normal = np.linalg.svd(xyz1 - centroid)[2][-1]
    if np.dot(xyz1[atom1] - centroid, normal) < 0:
        normal = -normal

    if len(xyz2) > 1:
        rest = np.delete(xyz2, atom2, axis=0).mean(axis=0) - xyz2[atom2]
        xyz2 = (xyz2 - xyz2[atom2]) @ _rotation(rest, normal).T + xyz2[atom2]
    xyz2 = xyz2 + (xyz1[atom1] + dist * normal - xyz2[atom2])

    return GeometryRow(
        symbols=geo1.symbols + geo2.symbols,
        coordinates=np.vstack([xyz1, xyz2]),
        charge=geo1.charge + geo2.charge,
        spin=geo1.spin + geo2.spin,
    )


def _rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the rotation matrix taking the direction of `a` onto that of `b`."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:  # noqa: PLR2004
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)
