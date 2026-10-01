"""Minimal ORCA runner (via OPI) returning results ready for autostorage.

Nothing here is specific to autostorage beyond building the `CalculationRow`;
any other program could be substituted.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from autostorage import (
    CalculationGeometryLink,
    CalculationRow,
    GeometryRow,
    ModelRow,
    Role,
)
from opi.core import Calculator
from opi.execution.core import Runner
from opi.input.structures import Structure
from opi.output.core import Output

_RESOURCES = {"memory": 8, "ncores": 1}


def configure(*, memory: int, ncores: int) -> None:
    """Set the memory (GB) and number of cores used by ORCA."""
    _RESOURCES.update(memory=memory, ncores=ncores)


@cache
def version() -> str:
    """Return the version of the ORCA binary."""
    return str(Runner().get_version())


@dataclass
class Result:
    """A finished ORCA calculation."""

    calculation: CalculationRow
    """The calculation, linked to its input geometry."""
    input: GeometryRow
    work_dir: Path
    basename: str
    output: Output

    def energy(self) -> float:
        """Final single-point energy (Eh)."""
        return _require(self.output.get_final_energy(), "energy")

    def gradient(self) -> np.ndarray:
        """Gradient (Eh/Bohr) of the last completed step."""
        return np.asarray(_require(self.output.get_gradient(index=-2), "gradient"))

    def zpe(self) -> float:
        """Zero-point vibrational energy (Eh)."""
        return _require(self.output.get_zpe(), "zero-point energy")

    def geometry(self) -> GeometryRow:
        """Final geometry, with the same charge and spin as the input."""
        return self._read_xyz(self.work_dir / f"{self.basename}.xyz")[0][0]

    def frames(self, filename: str) -> list[tuple[GeometryRow, float]]:
        """Geometries and energies from a multi-frame xyz file."""
        return self._read_xyz(self.work_dir / filename)

    def hessian(self) -> np.ndarray:
        """Cartesian Hessian (Eh/Bohr^2) from the `.hess` file."""
        lines = (self.work_dir / f"{self.basename}.hess").read_text().splitlines()
        start = lines.index("$hessian") + 1
        dim = int(lines[start])
        hess = np.zeros((dim, dim))
        row = start + 1
        # Stored in blocks of (up to) 5 columns, each preceded by a header line
        for col in range(0, dim, 5):
            for i in range(dim):
                values = lines[row + 1 + i].split()[1:]
                hess[i, col : col + len(values)] = values
            row += dim + 1
        return hess

    def _read_xyz(self, path: Path) -> list[tuple[GeometryRow, float]]:
        lines = [ln for ln in path.read_text().strip().splitlines() if ln != ">"]
        frames = []
        while lines:
            block, lines = lines[: int(lines[0]) + 2], lines[int(lines[0]) + 2 :]
            geo = GeometryRow.from_xyz_block(
                "\n".join(block), charge=self.input.charge, spin=self.input.spin
            )
            energy = re.search(r"-?\d+\.\d+", block[1])
            frames.append((geo, float(energy.group()) if energy else float("nan")))
        return frames


def run(
    geo: GeometryRow, model: ModelRow, calc_type: str, work_dir: Path, *blocks: str
) -> Result:
    """Run (or reuse a finished) ORCA calculation on `geo`.

    Args:
        geo: Input geometry.
        model: Level of theory.
        calc_type: ORCA job keyword (e.g. ``opt``, ``Freq``).
        work_dir: Directory for ORCA's files.
        blocks: Raw ORCA input blocks (e.g. ``%geom ... end``).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    keywords = [kw for kw in (model.method, model.basis, calc_type) if kw]

    calc = Calculator(basename=calc_type, working_dir=work_dir)
    output = calc.get_output()
    if not output.terminated_normally():
        calc.structure = Structure.from_xyz_block(
            geo.xyz_block(), charge=geo.charge, multiplicity=geo.spin + 1
        )
        # Convert GB to MiB, leaving headroom for ORCA's over-consumption
        calc.input.memory = int(_RESOURCES["memory"] * 953.674 * 0.75)
        calc.input.ncores = _RESOURCES["ncores"]
        calc.input.add_simple_keywords(*keywords)
        for block in blocks:
            calc.input.add_arbitrary_string(block)
        calc.write_input()
        calc.run()
        output = calc.get_output()

    output.parse()
    if not output.terminated_normally():
        msg = f"ORCA calculation failed, see {output.get_outfile()}"
        raise RuntimeError(msg)

    prop_json = work_dir / f"{calc_type}.property.json"
    properties = json.loads(prop_json.read_text()) if prop_json.exists() else {}
    properties.pop("Geometries", None)

    calculation = CalculationRow(
        model=model,
        calc_type=calc_type,
        input_provenance={"keywords": keywords, "blocks": list(blocks), **_RESOURCES},
        output_provenance=properties,
        geometry_links=[CalculationGeometryLink(geometry=geo, role=Role.INPUT)],
    )
    return Result(calculation, geo, work_dir, calc_type, output)


def _require[T](value: T | None, name: str) -> T:
    if value is None:
        msg = f"ORCA output is missing the {name}."
        raise ValueError(msg)
    return value
