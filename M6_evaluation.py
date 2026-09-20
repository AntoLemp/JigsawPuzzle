class Evaluator:
    def __init__(self, ground_truth_pos, ground_truth_rot, P, Q):
        self.gt_pos = ground_truth_pos
        self.gt_rot = ground_truth_rot
        self.P = P
        self.Q = Q
        self.N = P * Q

    def are_true_neighbors(self, piece_a, piece_b):
        row_a, col_a = self.gt_pos[piece_a]
        row_b, col_b = self.gt_pos[piece_b]

        return (
                abs(row_a - row_b)
                + abs(col_a - col_b)
                == 1
        )

    def evaluate(self, reconstructed_grid):
        """
        Computes quantitative metrics.
        """
        correct_placements = 0
        correct_rotations = 0

        # Check absolute placement and rotation accuracy
        for pos, (piece_idx, rot) in reconstructed_grid.items():
            gt_r, gt_c = self.gt_pos[piece_idx]
            # Adjusting for relative shifts might be necessary in a real scenario
            # assuming absolute grid alignment here.
            if pos == (gt_r, gt_c):
                correct_placements += 1
            if rot == self.gt_rot[piece_idx]:
                correct_rotations += 1

        placement_accuracy = correct_placements / self.N
        rotation_accuracy = correct_rotations / self.N

        return {
            "placement_accuracy": placement_accuracy,
            "rotation_accuracy": rotation_accuracy
        }

    def evaluate_neighbor_accuracy(self, reconstructed_grid):
        """
        Calculates the fraction of true neighboring pairs recovered.
        """
        reconstructed_pairs = set()

        for (r, c), (piece_idx, _) in reconstructed_grid.items():

            for neighbor_position in [
                (r + 1, c),
                (r, c + 1)
            ]:
                if neighbor_position not in reconstructed_grid:
                    continue

                neighbor_idx, _ = reconstructed_grid[
                    neighbor_position
                ]

                pair = tuple(sorted([
                    piece_idx,
                    neighbor_idx
                ]))

                reconstructed_pairs.add(pair)

        true_pairs = set()

        for piece_a in range(self.N):
            for piece_b in range(piece_a + 1, self.N):

                if self.are_true_neighbors(
                        piece_a,
                        piece_b
                ):
                    true_pairs.add(
                        (piece_a, piece_b)
                    )

        if not true_pairs:
            return 0.0

        recovered_pairs = (
            reconstructed_pairs.intersection(true_pairs)
        )

        return len(recovered_pairs) / len(true_pairs)