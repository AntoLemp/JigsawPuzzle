import cv2
import random

class PuzzleGeneration:
    def __init__(self, P, Q):

        self.P = P
        self.Q = Q
        self.ground_truth_positions = {}
        self.ground_truth_rotations = {}

    def generate_puzzle(self, image_path):
        image = cv2.imread(image_path)
        H, W, _ = image.shape

        h = H // self.P
        w = W // self.Q
        image = image[:h * self.P, :w * self.Q]

        pieces = []
        original_positions = {}
        piece_idx = 0

        for r in range(self.P):
            for c in range(self.Q):
                tile = image[r * h:(r + 1) * h, c * w:(c + 1) * w]
                pieces.append(tile)
                original_positions[piece_idx] = (r, c)
                piece_idx += 1

        N = self.P * self.Q
        shuffled_indices = list(range(N))
        random.shuffle(shuffled_indices)

        allowed_rotations = [0, 90, 180, 270]
        final_pieces = []

        for new_idx, orig_idx in enumerate(shuffled_indices):
            # This is the rotation used to scramble the original tile.
            scramble_angle = random.choice(allowed_rotations)

            tile = pieces[orig_idx]

            if scramble_angle == 90:
                tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            elif scramble_angle == 180:
                tile = cv2.rotate(tile, cv2.ROTATE_180)
            elif scramble_angle == 270:
                tile = cv2.rotate(tile, cv2.ROTATE_90_COUNTERCLOCKWISE)

            final_pieces.append(tile)

            # Map final index to the original (row, col) position
            self.ground_truth_positions[new_idx] = original_positions[orig_idx]

            # Store the rotation needed to undo the scrambling.
            inverse_angle = (-scramble_angle) % 360
            self.ground_truth_rotations[new_idx] = inverse_angle

        return final_pieces, h, w