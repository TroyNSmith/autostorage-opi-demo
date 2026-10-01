"""Custom identity algorithm and property kind registered with autostorage.

Anything registered here before a `Database` is opened is mirrored into it:
identities from `irmsd_conformer` are generated automatically for every new
stationary point, and `zpe` values are validated like any built-in property.
"""

import uuid
from collections.abc import Mapping

from automol import AlgorithmRegistry, Geometry, IdentityKind, rdkit_inchi
from autostorage import PropertyKindRegistry, energy_property_kind
from irmsd import Molecule, sorter_irmsd_molecule


def irmsd_identity_fn(
    geo: Geometry, other_geos: Mapping[str, Geometry] | None = None
) -> str:
    """Group conformers by iRMSD.

    `other_geos` holds one geometry per existing conformer identity among the
    stationary points sharing this geometry's parent identity (its InChI).
    """
    if not other_geos:
        return uuid.uuid4().hex

    confs = [Molecule(g.symbols, g.coordinates) for g in [*other_geos.values(), geo]]
    groups, _ = sorter_irmsd_molecule(confs, rthr=0.125)
    for key, group in zip(other_geos, groups[:-1], strict=True):
        if group == groups[-1]:
            return key
    return uuid.uuid4().hex


irmsd_conformer = AlgorithmRegistry.register(
    name="irmsd_conformer",
    kind=IdentityKind.CONFORMER,
    identity_fn=irmsd_identity_fn,
    parent_algorithm=rdkit_inchi,
)

zpe_property_kind = PropertyKindRegistry.register(
    name="zpe", validation_fn=energy_property_kind.validation_fn
)
