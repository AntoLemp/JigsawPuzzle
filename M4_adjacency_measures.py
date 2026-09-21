import cv2
import numpy as np
from scipy.spatial.distance import cosine

def cnn_boundary_score(feat_a, feat_b):
    """
    Compute a similarity score between two CNN feature vectors using cosine distance.
    :param feat_a: The first CNN feature vector.
    :param feat_b: The second CNN feature vector.
    :return: The similarity score.
    """
    dist = cosine(feat_a, feat_b)
    return float(np.exp(-dist / 0.5))

def rotate_piece(image, angle, target_height=None, target_width=None):
    """
    Rotate an image by a specified angle and optionally resize it to target dimensions.
    :param image: The image to rotate.
    :param angle: The angle by which to rotate the image.
    :param target_height: The target height for the resized image.
    :param target_width: The target width for the resized image.
    :return: The rotated and optionally resized image.
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
    """
    Validate that both tiles are valid images with three color channels.
    :param tile_a: The first tile.
    :param tile_b: The second tile.
    :return: None.
    """
    if tile_a is None or tile_b is None:
        raise ValueError("Both tiles must be valid images.")

    if tile_a.ndim != 3 or tile_b.ndim != 3:
        raise ValueError("Tiles must have shape H x W x 3.")

    if tile_a.shape[2] != 3 or tile_b.shape[2] != 3:
        raise ValueError("Tiles must have three color channels.")


def boundary_cost(tile_a, tile_b, direction, border_width=4):
    """
    Compute the average Lab color difference along the shared boundary of two tiles.
    :param tile_a: The first tile.
    :param tile_b: The second tile.
    :param direction: The direction of the boundary ("horizontal" or "vertical").
    :param border_width: The width of the boundary strip to consider.
    :return: The average Lab color difference.
    """
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
    Convert boundary cost into a score in approximately (0, 1].
    :param tile_a: The first tile.
    :param tile_b: The second tile.
    :param direction: The direction of the boundary ("horizontal" or "vertical").
    :param border_width: The width of the boundary strip to consider.
    :return: The boundary score.
    """
    cost = boundary_cost(
        tile_a,
        tile_b,
        direction,
        border_width=border_width
    )

    return float(np.exp(-cost / 20.0))


def gradient_boundary_cost(tile_a, tile_b, direction, border_width=4):
    """
    Compute the gradient boundary cost between two tiles using Sobel filters.
    :param tile_a: The first tile.
    :param tile_b: The second tile.
    :param direction: The direction of the boundary ("horizontal" or "vertical").
    :param border_width: The width of the boundary strip to consider.
    :return: The gradient boundary cost.
    """
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
    Convert gradient boundary cost into a score in approximately (0, 1].
    :param tile_a: The first tile.
    :param tile_b: The second tile.
    :param direction: The direction of the boundary ("horizontal" or "vertical").
    :param border_width: The width of the boundary strip to consider.
    :return: The gradient boundary score.
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
    Compute a combined boundary compatibility score between two tiles using color, texture, and CNN features.
    :param tile_a: The first tile.
    :param tile_b: The second tile.
    :param direction: The direction of the boundary ("horizontal" or "vertical").
    :param feat_a: The features for the first tile.
    :param feat_b: The features for the second tile.
    :param global_a: The global features for the first tile.
    :param global_b: The global features for the second tile.
    :param weights: The weights for combining the different feature types.
    :return: The combined boundary compatibility score.
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