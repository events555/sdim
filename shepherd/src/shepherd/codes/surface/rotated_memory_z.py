"""
rotated_memory_z.py -- the "rotated_memory_z" code: the rotated surface code on a d x d patch
(odd d), with its lattice geometry, SUM-gate schedule and syndrome-extraction patch circuit.

Work in progress. Implemented: the lattice (`generate_implicit_grid_rotated_z`), ancilla and
Hadamard indices, the four SUM layers (`sum_1` .. `sum_4`) and the patch circuit (`build_patch`,
returned by `encoding()`). Not implemented yet: `stabilizers()` (and so the default
`logical_operators()`), which raise NotImplementedError.
"""

from __future__ import annotations

import numpy as np
from sdim import Circuit

from shepherd.qecc import QECC, PauliString


class SurfaceCode(QECC, name="rotated_memory_z"):
    """
    Rotated surface code on a d x d lattice (odd d).
    Builds the lattice and the syndrome-extraction circuit on construction.
    """

    def __init__(self, dimension: int, distance: int, rounds: int = 1):
        if distance % 2 != 1:
            raise ValueError("Distance must be odd for the rotated surface code.")
        super().__init__(dimension, distance=distance, rounds=rounds)

        self.distance = distance
        self.rounds = rounds

        # Build adjacency graph and grid
        self.grid, self.adjacency = self.generate_implicit_grid_rotated_z()

        # Compute ancilla and Hadamard indices
        self.ancillas = self.ancilla_indices()
        self.hadamards = self.build_H_array()

        # Build the syndrome-extraction patch circuit
        self.encoding_circuits = [self.build_patch(dimension, rounds)]

    def stabilizers(self) -> list[PauliString]:
        raise NotImplementedError("rotated_memory_z: stabilizers are not implemented yet "
                                  "(they could be read off self.adjacency)")

    def encoding(self) -> Circuit:
        return self.encoding_circuits[0]

    def build_H_array(self):
        """
        Returns indices of ancilla qubits that require Hadamards
        in the surface code syndrome extraction circuit.
        Assumes odd distance.
        """
        d = self.distance
        result = []
        n = 0  # counter

        if (d % 2) == 1:
            # Hadamards on every ancilla in the top row
            head = (d - 1) // 2
            result.extend(range(0, head))
            n += head
            rows = 0

            # Take even numbers from even rows
            while rows < d - 1:
                # skip d
                n += d

                # take evens from next row
                block = np.arange(n, n + d)
                result.extend(block[1::2])

                n += d
                rows += 1

            n += d
            result.extend(range(n, n + head))

        else:
            # Even distance not implemented
            raise NotImplementedError("Even distance surface code not implemented.")

        return np.array(result)

    def sum_1(self):
        """
        Returns (controls, targets) for one round of CNOT scheduling
        in the surface code syndrome extraction.

        Assumes odd distance.
        """
        d = self.distance

        controls = []
        targets = []

        # index counter
        n = 0

        # Top row ancillas are always control
        head = (d - 1) // 2
        controls.extend(range(0, head))
        n += head

        # Even entries of the first row of data are targets
        block = range(n, n + d)
        targets.extend(block[1::2])
        n += d

        # Main pattern
        for _ in range(head):
            # --- Step 1 ---
            block = range(n, n + d)
            controls.extend(block[1::2])

            n += d
            block = range(n, n + d)
            targets.extend(block[2::2])

            # --- Step 2 ---
            controls.extend(block[1::2])

            block_prev = range(n - d, n)
            targets.extend(block_prev[0:-1:2])

            # --- Step 3 ---
            n += d
            block = range(n, n + d)
            controls.extend(block[1::2])

            n += d
            block = range(n, n + d)
            targets.extend(block[1::2])

            # --- Step 4 ---
            controls.extend(block[0::2])

            block_prev = range(n - d, n)
            targets.extend(block_prev[0::2])

            n += d

        return (np.array(controls), np.array(targets))

    def sum_2(self):
        """
        Returns (controls, targets) for the second CNOT layer
        in the surface code syndrome extraction schedule.
        """
        d = self.distance

        ctrl = []
        targ = []

        if d % 2 != 0:
            # --- Odd distance ---
            num_outer_ancilla = (d - 1) // 2

            # Top ancillae
            for j in range(num_outer_ancilla):
                ctrl.append(j)
                targ.append((2 * j) + num_outer_ancilla)

            # Main block
            on_ancilla_line = False
            ancilla_leading_left = True

            for row in range((2 * d) - 2):

                if on_ancilla_line:
                    ancilla_leading_left = not ancilla_leading_left

                data_start_offset = 1 if ancilla_leading_left else 0
                start_index = 1 if on_ancilla_line else data_start_offset
                target_offset = d - 1 if ancilla_leading_left else d

                for j in range(start_index, d, 2):
                    base = num_outer_ancilla + (row * d)
                    ctrl.append(base + j)
                    targ.append(base + target_offset + j)

                on_ancilla_line = not on_ancilla_line

        else:
            # --- Even distance (still marked TODO in your logic) ---
            num_outer_ancilla = d // 2

            # Top ancillae
            for j in range(num_outer_ancilla):
                ctrl.append(j)
                targ.append((2 * j) + num_outer_ancilla)

            # Main block
            on_ancilla_line = False
            ancilla_leading = True

            for row in range((2 * d) - 2):

                row_end = d

                if on_ancilla_line:
                    ancilla_leading = not ancilla_leading
                    row_end = d + 1 if ancilla_leading else d - 1

                # ⚠️ FIX: use ancilla_leading (not ancilla_leading_left)
                data_start_offset = 1 if ancilla_leading else 0
                start_index = 1 if on_ancilla_line else data_start_offset
                target_offset = d - 1 if ancilla_leading else d

                for j in range(start_index, row_end, 2):
                    base = num_outer_ancilla + (row * d)
                    ctrl.append(base + j)
                    targ.append(base + target_offset + j)

                on_ancilla_line = not on_ancilla_line

        return (np.array(ctrl), np.array(targ))

    def sum_3(self):
        """
        Third CNOT layer.
        Mirror of SUM_2 via index reversal.
        """
        d = self.distance

        total = 2 * (d**2 - 1)

        ctrl, targ = self.sum_2()

        ctrl = total - ctrl
        targ = total - targ

        return (ctrl, targ)

    def sum_4(self):
        """
        Fourth CNOT layer.
        Mirror of SUM_1 (reversed indexing).
        """
        d = self.distance

        C = 2 * (d**2 - 1)

        ctrl, targ = self.sum_1()

        ctrl = C - ctrl
        targ = C - targ

        return (ctrl, targ)

    def ancilla_indices(self):
        """
        Returns the indices of all ancilla qubits in the lattice.
        Assumes odd distance.
        """
        d = self.distance

        index = []
        n = 0
        head = (d - 1) // 2

        # Top ancillas
        index.extend(range(0, head))
        n += head

        # Bulk ancilla rows
        for _ in range(d - 1):
            n += d
            block = range(n, n + d)
            index.extend(block)
            n += d

        # Bottom ancillas
        n += d
        index.extend(range(n, n + head))

        return index

    def generate_implicit_grid_rotated_z(self):
        """
        Creates a grid + adjacency list for a rotated Z surface code patch.

        Returns:
            grid: dict[int -> (x, y)]
            adjacency_list: dict[int -> dict[int -> str]]
        """
        d = self.distance

        grid = {}
        adjacency_list = {}

        if d % 2 != 0:
            length = 2 * d + 1
            num_outer_ancilla = (d - 1) // 2
            total = (d * (2 * d - 1)) + (2 * num_outer_ancilla)

            qudit_index = 0

            # Initialize adjacency list
            for j in range(total):
                adjacency_list[j] = {}

            # ------------------
            # Top ancillae
            # ------------------
            for _ in range(num_outer_ancilla):
                grid[qudit_index] = (2 + (4 * qudit_index), length - 1)

                adjacency_list[qudit_index][qudit_index + num_outer_ancilla] = "W"
                adjacency_list[qudit_index + num_outer_ancilla][qudit_index] = "E"

                adjacency_list[qudit_index][qudit_index + num_outer_ancilla + 1] = "S"
                adjacency_list[qudit_index + num_outer_ancilla + 1][qudit_index] = "N"

                qudit_index += 1

            # ------------------
            # Body
            # ------------------
            for row in range(1, length - 1):

                if row % 2 == 0:
                    # ===== ANCILLA ROW =====
                    ancilla_leading_left = (row % 4 == 0)
                    next_ancilla_leading_left = ((row + 2) % 4 == 0)  # ✅ FIX

                    start_qudit_index = qudit_index

                    if ancilla_leading_left:
                        grid[qudit_index] = (0, length - (row + 1))
                        qudit_index += 1

                    for col in range(2, length - 1, 2):
                        grid[qudit_index] = (col, length - (row + 1))
                        qudit_index += 1

                    if not ancilla_leading_left:
                        grid[qudit_index] = (length - 1, length - (row + 1))
                        qudit_index += 1

                    end_qudit_index = qudit_index

                    edge_case_index = start_qudit_index if ancilla_leading_left else end_qudit_index - 1

                    edge_case_down = "S" if next_ancilla_leading_left else "W"
                    edge_case_up = "N" if next_ancilla_leading_left else "E"

                    lookahead_offset = d - 1 if ancilla_leading_left else d

                    for index in range(start_qudit_index, end_qudit_index):

                        if index == edge_case_index:
                            adjacency_list[index][index + d] = edge_case_down
                            adjacency_list[index + d][index] = edge_case_up
                        else:
                            adjacency_list[index][index + lookahead_offset] = "W"
                            adjacency_list[index + lookahead_offset][index] = "E"

                            adjacency_list[index][index + lookahead_offset + 1] = "S"
                            adjacency_list[index + lookahead_offset + 1][index] = "N"

                else:
                    # ===== DATA ROW =====
                    start_qudit_index = qudit_index

                    for col in range(1, length, 2):
                        grid[qudit_index] = (col, length - (row + 1))
                        qudit_index += 1

                    end_qudit_index = qudit_index

                    if row < length - 2:
                        next_ancilla_leading_left = ((row + 1) % 4 == 0)

                        edge_case_index = end_qudit_index - 1 if next_ancilla_leading_left else start_qudit_index

                        edge_case_down = "W" if next_ancilla_leading_left else "S"
                        edge_case_up = "E" if next_ancilla_leading_left else "N"

                        lookahead_offset = d if next_ancilla_leading_left else d - 1

                        for index in range(start_qudit_index, end_qudit_index):

                            if index == edge_case_index:
                                adjacency_list[index][index + d] = edge_case_down
                                adjacency_list[index + d][index] = edge_case_up
                            else:
                                adjacency_list[index][index + lookahead_offset] = "W"
                                adjacency_list[index + lookahead_offset][index] = "E"

                                adjacency_list[index][index + lookahead_offset + 1] = "S"
                                adjacency_list[index + lookahead_offset + 1][index] = "N"

            # ------------------
            # Bottom ancillae
            # ------------------
            final_row_index = qudit_index

            for j in range(num_outer_ancilla):
                grid[qudit_index] = (4 + (4 * j), 0)

                jj = num_outer_ancilla - (j + 1)

                left_upper_index = final_row_index - ((2 * jj) + 2)
                right_upper_index = final_row_index - ((2 * jj) + 1)

                adjacency_list[qudit_index][left_upper_index] = "N"
                adjacency_list[left_upper_index][qudit_index] = "S"

                adjacency_list[qudit_index][right_upper_index] = "E"
                adjacency_list[right_upper_index][qudit_index] = "W"

                qudit_index += 1

        else:
            raise NotImplementedError("Even distance not implemented.")

        return grid, adjacency_list

    def build_patch(self, q: int, rounds: int):
        """
        Builds the surface code circuit for given qudit dimension q
        and number of syndrome extraction rounds.
        """
        d = self.distance

        if d % 2 == 0:
            raise ValueError("distance should be odd")

        total = 2 * (d**2) - 1

        patch = Circuit(num_qudits=total, dimension=q)

        # ------------------
        # Encoding step
        # ------------------
        patch.add_gate("H", self.hadamards)

        for a in self.ancillas :
            if a in self.hadamards:
                for k in self.adjacency[a]:
                    if self.adjacency[a][k] in ("N", "E"):
                        patch.add_gate("SUM_INV", int(a), int(k))
                    else:
                        patch.add_gate("SUM", int(a), int(k))
            else:
                for k in self.adjacency[a]:
                    if self.adjacency[a][k] in ("E", "S"):
                        patch.add_gate("SUM_INV", int(k), int(a))
                    else:
                        patch.add_gate("SUM", int(k), int(a))

        patch.add_gate("H_INV", self.hadamards)
        patch.add_gate("MR", self.ancillas )

        # ------------------
        # Syndrome extraction rounds
        # ------------------
        for _ in range(rounds):

            patch.add_gate("H", self.hadamards)

            for a in self.ancillas:
                if a in self.hadamards:
                    for k in self.adjacency[a]:
                        if self.adjacency[a][k] in ("N", "E"):
                            patch.add_gate("SUM_INV", int(a), int(k))
                        else:
                            patch.add_gate("SUM", int(a), int(k))
                else:
                    for k in self.adjacency[a]:
                        if self.adjacency[a][k] in ("E", "S"):
                            patch.add_gate("SUM_INV", int(k), int(a))
                        else:
                            patch.add_gate("SUM", int(k), int(a))

            patch.add_gate("H_INV", self.hadamards)
            patch.add_gate("MR", self.ancillas )

        return patch
