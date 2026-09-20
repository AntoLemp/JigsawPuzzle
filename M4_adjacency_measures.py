import cv2
import numpy as np
from scipy.spatial.distance import cosine

def cnn_boundary_score(feat_a, feat_b):
    dist = cosine(feat_a, feat_b)
    return float(np.exp(-dist / 0.5))

def rotate_piece(image, angle, target_height=None, target_width=None):
    """
    Rotate a piece and optionally resize it to a common shape.

    This is useful when original tiles are rectangular because 90-degree
    rotations swap height and width.
    """
    angle = angle % 360

    if angle == 90:
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )
    elif angle == 180:
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_180
        )
    elif angle == 270:
        rotated = cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )
    else:
        rotated = image.copy()

    if target_height is not None and target_width is not None:
        rotated = cv2.resize(
            rotated,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR
        )

    return rotated


def _validate_tiles(tile_a, tile_b):
    if tile_a is None or tile_b is None:
        raise ValueError("Both tiles must be valid images.")

    if tile_a.ndim != 3 or tile_b.ndim != 3:
        raise ValueError("Tiles must have shape H x W x 3.")

    if tile_a.shape[2] != 3 or tile_b.shape[2] != 3:
        raise ValueError("Tiles must have three color channels.")


def boundary_cost(tile_a, tile_b, direction, border_width=4):
    """Compute direct Lab color mismatch across a shared boundary by cropping first."""
    _validate_tiles(tile_a, tile_b)

    if direction == "horizontal":
        strip_a = tile_a[:, -border_width:, :]
        strip_b = tile_b[:, :border_width, :]
    else:
        strip_a = tile_a[-border_width:, :, :]
        strip_b = tile_b[:border_width, :, :]

    lab_a = cv2.cvtColor(strip_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(strip_b, cv2.COLOR_BGR2LAB).astype(np.float32)

    differences = []
    if direction == "horizontal":
        for k in range(border_width):
            differences.append(np.mean(np.abs(lab_a[:, -(k + 1), :] - lab_b[:, k, :])))
    else:
        for k in range(border_width):
            differences.append(np.mean(np.abs(lab_a[-(k + 1), :, :] - lab_b[k, :, :])))

    return float(np.mean(differences))


def boundary_score(tile_a, tile_b, direction, border_width=4):
    """
    Convert Lab boundary cost into a score in approximately (0, 1].

    Higher values are better.
    """
    cost = boundary_cost(
        tile_a,
        tile_b,
        direction,
        border_width=border_width
    )

    return float(np.exp(-cost / 20.0))


def gradient_boundary_cost(tile_a, tile_b, direction, border_width=4):
    """Compare Sobel responses near the shared boundary by cropping first."""
    _validate_tiles(tile_a, tile_b)
    pad = border_width + 2

    if direction == "horizontal":
        strip_a = tile_a[:, -min(pad, tile_a.shape[1]):]
        strip_b = tile_b[:, :min(pad, tile_b.shape[1])]
    else:
        strip_a = tile_a[-min(pad, tile_a.shape[0]):, :]
        strip_b = tile_b[:min(pad, tile_b.shape[0]), :]

    gray_a = cv2.cvtColor(strip_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(strip_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx_a = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
    gy_a = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)
    gx_b = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
    gy_b = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)

    if direction == "horizontal":
        gx_cost = np.mean(np.abs(gx_a[:, -border_width:] - gx_b[:, :border_width]))
        gy_cost = np.mean(np.abs(gy_a[:, -border_width:] - gy_b[:, :border_width]))
    else:
        gx_cost = np.mean(np.abs(gx_a[-border_width:, :] - gx_b[:border_width, :]))
        gy_cost = np.mean(np.abs(gy_a[-border_width:, :] - gy_b[:border_width, :]))

    return float(gx_cost + gy_cost)


def gradient_boundary_score(tile_a, tile_b, direction, border_width=4):
    """
    Convert gradient cost into a score in approximately (0, 1].
    """
    cost = gradient_boundary_cost(
        tile_a,
        tile_b,
        direction,
        border_width=border_width
    )

    # Sobel values can be much larger than Lab differences.
    return float(np.exp(-cost / 300.0))

def boundary_compatibility(tile_a, tile_b, direction, feat_a, feat_b,
                           global_a, global_b, weights):
    """
        Combined direct boundary score.

        Higher values are better.
        """
    color_side = boundary_score(tile_a, tile_b, direction)
    texture_side = gradient_boundary_score(tile_a, tile_b, direction)
    cnn_side = cnn_boundary_score(feat_a['cnn'], feat_b['cnn'])

    cnn_tile = cnn_boundary_score(global_a['cnn'], global_b['cnn'])

    side_score = (weights['color'] * color_side +
                  weights['texture'] * texture_side +
                  weights['cnn'] * cnn_side)

    tile_score = weights['cnn_tile'] * cnn_tile

    return (weights['w_side'] * side_score) + (weights['w_tile'] * tile_score)