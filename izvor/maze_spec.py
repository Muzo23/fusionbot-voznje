"""
Maze specifications (step 9; the CoppeliaSim scene generator follows in step 12).

A maze is a topological graph on a square lattice with pitch PITCH_M. Every
node is a free square of side FREE_WIDTH_M, every edge a free corridor of the
same width. A separate START approach corridor (not part of the graph) leads
north into the ROOT node; the robot starts on it START_OFFSET_FROM_ROOT_M south
of the ROOT centre, heading NORTH.

The geometry constants and the T1/T2/T3 graphs are identical to the generator
that built the CoppeliaSim scenes used so far (FusionBot2), so the offline
walls match the simulator walls exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PITCH_M = 1.085268
FREE_WIDTH_M = 0.728712
HALF_FREE_WIDTH_M = 0.5 * FREE_WIDTH_M
WALL_THICKNESS_M = 0.05
START_OFFSET_FROM_ROOT_M = 1.422634
START_BACK_CLEARANCE_M = 0.60
START_HEADING_RAD = 0.5 * math.pi          # NORTH


@dataclass(frozen=True)
class MazeSpec:
    maze_id: str
    name: str
    nodes_grid: dict[str, tuple[int, int]]
    edges: tuple[tuple[str, str], ...]
    root: str
    goal: str
    expected_cycle_count: int

    @property
    def nodes_m(self) -> dict[str, tuple[float, float]]:
        return {n: (gx * PITCH_M, gy * PITCH_M) for n, (gx, gy) in self.nodes_grid.items()}

    @property
    def start_pose(self) -> tuple[float, float, float]:
        rx, ry = self.nodes_m[self.root]
        return rx, ry - START_OFFSET_FROM_ROOT_M, START_HEADING_RAD

    def neighbours(self, node: str) -> list[str]:
        out = [b for a, b in self.edges if a == node] + [a for a, b in self.edges if b == node]
        return sorted(out)


MAZES: dict[str, MazeSpec] = {
    "T1": MazeSpec(
        "T1", "Turn / U-Turn Stress Tree",
        {"R": (0, 0), "A": (0, 2), "B": (-2, 2), "C": (-2, 3), "D": (-3, 3), "E": (-3, 5),
         "F": (-5, 5), "G": (-3, 6), "H": (-2, 6), "S": (1, 0), "Q": (1, 2), "P": (3, 2),
         "N": (3, 3), "M": (3, 5), "K": (1, 5), "J": (0, 5), "I": (0, 3), "L": (1, 6),
         "O": (4, 3), "T": (4, 2), "U": (-2, 0), "V": (-2, -1)},
        (("R", "A"), ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("E", "G"),
         ("G", "H"), ("R", "S"), ("S", "Q"), ("Q", "P"), ("P", "N"), ("N", "M"), ("M", "K"),
         ("K", "J"), ("J", "I"), ("K", "L"), ("N", "O"), ("P", "T"), ("R", "U"), ("U", "V")),
        "R", "F", 0,
    ),
    "T2": MazeSpec(
        "T2", "Loop / Revisit Stress",
        {"R": (0, 0), "A": (0, 3), "D": (2, 3), "B": (4, 3), "C": (4, 0), "E": (2, 0),
         "H": (2, 4), "G": (5, 3), "F": (-1, 0)},
        (("R", "A"), ("A", "D"), ("D", "B"), ("B", "C"), ("C", "E"), ("E", "R"), ("D", "E"),
         ("D", "H"), ("B", "G"), ("R", "F")),
        "R", "G", 2,
    ),
    "T3": MazeSpec(
        "T3", "Long-Corridor Drift Ring",
        {"R": (0, 0), "A": (0, 5), "B": (5, 5), "C": (5, 0), "E": (-2, 0), "F": (5, 7),
         "G": (7, 0)},
        (("R", "A"), ("A", "B"), ("B", "C"), ("C", "R"), ("R", "E"), ("B", "F"), ("C", "G")),
        "R", "F", 1,
    ),
}


def _register_real_mazes() -> None:
    """M1..M7: real micromouse mazes (data/mazes, converted by mazes.micromouse)."""
    from mazes.micromouse import REAL_MAZES, real_maze
    for mid in REAL_MAZES:
        try:
            MAZES[mid] = real_maze(mid)
        except FileNotFoundError:          # data folder not shipped
            pass


_register_real_mazes()
