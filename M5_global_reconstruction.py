import random
import math
import numpy as np
from M4_adjacency_measures import (
    rotate_piece,
    boundary_compatibility
)


class Reconstructor:
    def __init__(self, P, Q, pieces, piece_features, weights, border_width=4):
        self.P = P
        self.Q = Q
        self.pieces = pieces
        self.piece_features = piece_features
        self.weights = weights
        self.border_width = border_width
        self.tile_height = pieces[0].shape[0]
        self.tile_width = pieces[0].shape[1]
        self.score_cache = {}
        self._precompute_scores()

    def _precompute_scores(self):
        orientations = [0, 90, 180, 270]
        directions = ["horizontal", "vertical"]

        for p_a in range(self.P * self.Q):
            for o_a in orientations:
                tile_a = self._get_rotated_piece(p_a, o_a)

                for p_b in range(self.P * self.Q):
                    if p_a == p_b:
                        continue

                    for o_b in orientations:
                        tile_b = self._get_rotated_piece(p_b, o_b)

                        for d in directions:
                            if d == "horizontal":
                                side_a, side_b = 'right', 'left'
                            else:
                                side_a, side_b = 'bottom', 'top'

                            feat_a = {
                                'cnn': self._get_active_feature(p_a, o_a, side_a, 'cnn'),
                                'texture': self._get_active_feature(p_a, o_a, side_a, 'texture')
                            }
                            feat_b = {
                                'cnn': self._get_active_feature(p_b, o_b, side_b, 'cnn'),
                                'texture': self._get_active_feature(p_b, o_b, side_b, 'texture')
                            }

                            global_a = self.piece_features[p_a]['tile']
                            global_b = self.piece_features[p_b]['tile']

                            # Calculate the score once and save it
                            score = boundary_compatibility(
                                tile_a, tile_b, d,
                                feat_a, feat_b,
                                global_a, global_b,
                                self.weights
                            )
                            self.score_cache[(p_a, o_a, p_b, o_b, d)] = score

    def _get_rotated_piece(self, piece_idx, orientation):
        return rotate_piece(
            self.pieces[piece_idx],
            orientation,
            target_height=self.tile_height,
            target_width=self.tile_width
        )

    def _pair_score(self, piece_a, orientation_a, piece_b, orientation_b, direction):
        """
        Retrieve the precomputed score instantly.
        """
        if piece_a == piece_b:
            return -np.inf

        return self.score_cache[(piece_a, orientation_a, piece_b, orientation_b, direction)]

    def greedy_reconstruction(self):
        """
        Build a fixed P x Q grid in row-major order.

        This is a baseline, not a globally optimal solver.
        """
        N = self.P * self.Q
        unassigned = set(range(N))

        # Try all possible first pieces and orientations.
        best_grid = None
        best_score = -np.inf

        for seed_piece in range(N):
            for seed_orientation in [0, 90, 180, 270]:

                candidate_grid = {
                    (0, 0): (seed_piece, seed_orientation)
                }

                remaining = set(unassigned)
                remaining.remove(seed_piece)

                for r in range(self.P):
                    for c in range(self.Q):
                        if (r, c) == (0, 0):
                            continue

                        best_move = None
                        best_move_score = -np.inf

                        for piece_idx in remaining:
                            for orientation in [0, 90, 180, 270]:
                                score = self._evaluate_placement(
                                    (r, c),
                                    piece_idx,
                                    orientation,
                                    candidate_grid
                                )

                                if score > best_move_score:
                                    best_move_score = score
                                    best_move = (
                                        piece_idx,
                                        orientation
                                    )

                        piece_idx, orientation = best_move
                        candidate_grid[(r, c)] = (
                            piece_idx,
                            orientation
                        )
                        remaining.remove(piece_idx)

                candidate_score = self._calculate_total_score(candidate_grid)

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_grid = candidate_grid

        return best_grid

    def _realign_grid(self, grid):
        """
        Shifts the reconstructed grid so that the top-left-most piece
        is at (0, 0), aligning it with the absolute ground-truth coordinate system.
        """
        if not grid:
            return grid

        # Find the boundaries of our floating puzzle
        min_r = min(r for r, c in grid.keys())
        min_c = min(c for r, c in grid.keys())

        aligned_grid = {}
        for (r, c), (piece_idx, orientation) in grid.items():
            # Shift everything back to start at (0, 0)
            aligned_grid[(r - min_r, c - min_c)] = (piece_idx, orientation)

        return aligned_grid

    def _get_open_neighbors(self, grid):
        open_spots = set()
        for (r, c) in grid.keys():
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for nr, nc in neighbors:
                # We no longer hard-bound to 0 <= nr < P,
                # because the grid is floating. We use our validity check.
                if (nr, nc) not in grid and self._is_valid_spot((nr, nc), grid):
                    open_spots.add((nr, nc))
        return open_spots

    def _get_active_feature(self, piece_idx, orientation, facing_side, feature_type):
        mapping = {
            0: {'top': 'top', 'right': 'right', 'bottom': 'bottom', 'left': 'left'},
            90: {'top': 'left', 'right': 'top', 'bottom': 'right', 'left': 'bottom'},
            180: {'top': 'bottom', 'right': 'left', 'bottom': 'top', 'left': 'right'},
            270: {'top': 'right', 'right': 'bottom', 'bottom': 'left', 'left': 'top'}
        }
        original_side = mapping[orientation][facing_side]
        return self.piece_features[piece_idx]['side'][feature_type][original_side]

    def _evaluate_placement(
            self,
            spot,
            piece_idx,
            orientation,
            grid
    ):
        """
        Evaluate a candidate placement using direct pixel compatibility.
        """
        score = 0.0
        r, c = spot

        # Candidate is below the top neighbor.
        if (r - 1, c) in grid:
            neighbor_idx, neighbor_orientation = grid[(r - 1, c)]

            score += self._pair_score(
                piece_a=neighbor_idx,
                orientation_a=neighbor_orientation,
                piece_b=piece_idx,
                orientation_b=orientation,
                direction="vertical"
            )

        # Candidate is above the bottom neighbor.
        if (r + 1, c) in grid:
            neighbor_idx, neighbor_orientation = grid[(r + 1, c)]

            score += self._pair_score(
                piece_a=piece_idx,
                orientation_a=orientation,
                piece_b=neighbor_idx,
                orientation_b=neighbor_orientation,
                direction="vertical"
            )

        # Candidate is right of the left neighbor.
        if (r, c - 1) in grid:
            neighbor_idx, neighbor_orientation = grid[(r, c - 1)]

            score += self._pair_score(
                piece_a=neighbor_idx,
                orientation_a=neighbor_orientation,
                piece_b=piece_idx,
                orientation_b=orientation,
                direction="horizontal"
            )

        # Candidate is left of the right neighbor.
        if (r, c + 1) in grid:
            neighbor_idx, neighbor_orientation = grid[(r, c + 1)]

            score += self._pair_score(
                piece_a=piece_idx,
                orientation_a=orientation,
                piece_b=neighbor_idx,
                orientation_b=neighbor_orientation,
                direction="horizontal"
            )

        return score

    def _is_valid_spot(self, spot, grid):
        """
        Ensures that placing a piece at 'spot' does not cause the
        reconstructed grid to exceed the strict P x Q dimensions.
        """
        if not grid:
            return True

        r_new, c_new = spot

        # Get current boundaries including the proposed spot
        min_r = min(r_new, min(r for r, c in grid.keys()))
        max_r = max(r_new, max(r for r, c in grid.keys()))

        min_c = min(c_new, min(c for r, c in grid.keys()))
        max_c = max(c_new, max(c for r, c in grid.keys()))

        # Check if the new bounding box exceeds P (height) or Q (width)
        height = (max_r - min_r) + 1
        width = (max_c - min_c) + 1

        return height <= self.P and width <= self.Q

    def _calculate_total_score(self, grid):
        """
        Calculate the global adjacency goal.

        Each horizontal and vertical adjacency is counted once.
        """
        total_score = 0.0

        for (r, c), (piece_idx, orientation) in grid.items():

            if (r, c + 1) in grid:
                right_idx, right_orientation = grid[
                    (r, c + 1)
                ]

                total_score += self._pair_score(
                    piece_a=piece_idx,
                    orientation_a=orientation,
                    piece_b=right_idx,
                    orientation_b=right_orientation,
                    direction="horizontal"
                )

            if (r + 1, c) in grid:
                bottom_idx, bottom_orientation = grid[
                    (r + 1, c)
                ]

                total_score += self._pair_score(
                    piece_a=piece_idx,
                    orientation_a=orientation,
                    piece_b=bottom_idx,
                    orientation_b=bottom_orientation,
                    direction="vertical"
                )

        return total_score

    def optimize_grid(self, grid, iterations=5000):
        """
        Simulated annealing over piece swaps and orientations.
        """
        current_grid = dict(grid)
        current_score = self._calculate_total_score(current_grid)

        best_grid = dict(current_grid)
        best_score = current_score

        positions = list(current_grid.keys())

        for iteration in range(iterations):
            candidate_grid = dict(current_grid)
            move_type = random.choice(["swap", "rotate"])

            if move_type == "swap":
                pos_a, pos_b = random.sample(positions, 2)

                candidate_grid[pos_a], candidate_grid[pos_b] = (
                    candidate_grid[pos_b],
                    candidate_grid[pos_a]
                )

            else:
                position = random.choice(positions)
                piece_idx, old_orientation = candidate_grid[position]

                possible_rotations = [
                    angle for angle in [0, 90, 180, 270]
                    if angle != old_orientation
                ]

                new_orientation = random.choice(possible_rotations)

                candidate_grid[position] = (
                    piece_idx,
                    new_orientation
                )

            candidate_score = self._calculate_total_score(candidate_grid)
            delta = candidate_score - current_score

            temperature = max(
                0.01,
                1.0 - iteration / iterations
            )

            accept = (
                    delta > 0
                    or random.random() < math.exp(delta / temperature)
            )

            if accept:
                current_grid = candidate_grid
                current_score = candidate_score

            if current_score > best_score:
                best_grid = dict(current_grid)
                best_score = current_score

        return best_grid

    def align_global_orientation(self, reconstructed_grid, evaluator):
        best_grid = reconstructed_grid
        best_metrics = evaluator.evaluate(reconstructed_grid)

        current_grid = reconstructed_grid
        for _ in range(3):
            next_grid = {}

            # Calculate current dimensions for coordinate rotation
            min_r = min(r for r, c in current_grid.keys())
            max_r = max(r for r, c in current_grid.keys())
            height = max_r - min_r + 1

            # Rotate the entire grid 90 degrees clockwise
            for (r, c), (piece_idx, orientation) in current_grid.items():
                new_r = c
                new_c = height - 1 - r
                new_orient = (orientation + 90) % 360
                next_grid[(new_r, new_c)] = (piece_idx, new_orient)

            # Shift the grid back to absolute (0,0) origin
            min_r_new = min(r for r, c in next_grid.keys())
            min_c_new = min(c for r, c in next_grid.keys())
            current_grid = {
                (r - min_r_new, c - min_c_new): (pid, ort)
                for (r, c), (pid, ort) in next_grid.items()
            }

            # Evaluate this global rotation
            metrics = evaluator.evaluate(current_grid)
            if metrics["placement_accuracy"] > best_metrics["placement_accuracy"]:
                best_metrics = metrics
                best_grid = current_grid

        return best_grid