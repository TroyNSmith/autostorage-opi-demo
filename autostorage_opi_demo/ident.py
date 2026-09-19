"""iRMSD conformer identity algorithm."""

from pathlib import Path

from automol.geom import Geometry
from automol.ident import AlgorithmFns, AlgorithmRegistry, Identity, IdentityKind
from irmsd import Molecule, read_structures, sorter_irmsd_molecule

IRMSD_CONFORMER = "irmsd_conformer"


@AlgorithmRegistry.register(IRMSD_CONFORMER, IdentityKind.CONFORMER)
class IrmsdConformerIdentity(AlgorithmFns):
    """iRMSD conformer identity."""

    @staticmethod
    def identity_fn(
        geo: Geometry, other_geos: dict[str, Geometry] | None = None
    ) -> str:
        """Generate conformer identity from Geometry referencing other Geometries."""
        if not hasattr(geo, "id"):
            msg = "Geometry object must have an 'id' attribute for conformer identity."
            raise ValueError(msg)

        if not other_geos:
            return str(geo.id)

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

        return str(geo.id)
