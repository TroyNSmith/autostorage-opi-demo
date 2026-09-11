"""iRMSD conformer identity algorithm."""

from pathlib import Path

from automol import ident
from irmsd import read_structures, sorter_irmsd_molecule

GOAT_DIR = Path(__file__).parent.parent / "out/1_GOAT"

confs = read_structures(str(GOAT_DIR / "test.xyz"))
groups, _ = sorter_irmsd_molecule(confs, rthr=0.125)

IRMSD_CONFORMER = ident.Algorithm("irmsd", "conformer")
print(IRMSD_CONFORMER)
