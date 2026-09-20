import matplotlib.pyplot as plt
import cv2


class Visualizer:
    def __init__(self, P, Q):
        self.P = P
        self.Q = Q

    def _rotate_image(self, image, angle):
        """Helper to rotate the image for display."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def show_shuffled(self, pieces, title="Shuffled Pieces"):
        """Displays the pieces as they are handed to the algorithm."""
        fig, axes = plt.subplots(self.P, self.Q, figsize=(8, 8))
        fig.suptitle(title, fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i < len(pieces):
                img_rgb = cv2.cvtColor(pieces[i], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def show_reconstruction(self, pieces, grid, title="Reconstructed Puzzle"):
        """Displays the final grid layout with applied rotations."""
        fig, axes = plt.subplots(self.P, self.Q, figsize=(8, 8))
        fig.suptitle(title, fontsize=16)

        for r in range(self.P):
            for c in range(self.Q):
                ax = axes[r, c]
                if (r, c) in grid:
                    piece_idx, orientation = grid[(r, c)]

                    # Rotate back to the estimated orientation
                    img_rotated = self._rotate_image(pieces[piece_idx], orientation)
                    img_rgb = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2RGB)

                    ax.imshow(img_rgb)
                ax.axis('off')

        plt.tight_layout()
        plt.show()