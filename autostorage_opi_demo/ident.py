"""iRMSD conformer identity algorithm."""

import uuid
from collections.abc import Mapping

from automol import rdkit_inchi
from automol.geom import Geometry
from automol.ident import AlgorithmRegistry, IdentityKind
from irmsd import Molecule, sorter_irmsd_molecule


def irmsd_identity_fn(
    geo: Geometry, other_geos: Mapping[str, Geometry] | None = None
) -> str:
    """iRMSD-based conformer grouping."""
    if not other_geos:
        return uuid.uuid4().hex

    keys = list(other_geos.keys())
    confs = [
        Molecule(g.symbols, g.coordinates)
        for g in [*list(other_geos.values() or []), geo]
    ]

    groups, _ = sorter_irmsd_molecule(confs, rthr=0.125)
    geo_group = groups[-1]
    for key, group in zip(keys, groups[:-1], strict=True):
        if group == geo_group:
            return key

    return uuid.uuid4().hex


irmsd_confomer = AlgorithmRegistry.register(
    name="irmsd_conformer",
    kind=IdentityKind.CONFORMER,
    identity_fn=irmsd_identity_fn,
    parent_algorithm=rdkit_inchi,
)
