import PySimpleGUI as sg
import cv2
import numpy as np
import time

from M1_puzzle_generation import PuzzleGeneration
from M2_3_feature_extraction import FeatureExtractor
from M5_global_reconstruction import Reconstructor
from M6_evaluation import Evaluator


def rotate_cv2_image(image, angle):
    """
    Rotates an OpenCV image by the specified angle (0, 90, 180, 270 degrees).
    :param image: The input image.
    :param angle: The rotation angle.
    :return: The rotated image.
    """
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def render_grid_image(P, Q, pieces, grid, gt_pos=None, gt_rot=None, highlight=False):
    """
    Renders the puzzle pieces on a grid based on their positions and orientations.
    :param P: The number of rows.
    :param Q: The number of columns.
    :param pieces: The list of puzzle pieces.
    :param grid: The grid layout mapping positions to piece indices and orientations.
    :param gt_pos: The ground truth positions.
    :param gt_rot: The ground truth rotations.
    :param highlight: Whether to highlight errors.
    :return: The rendered grid image.
    """

    if not pieces:
        return np.zeros((400, 400, 3), dtype=np.uint8)

    h, w = pieces[0].shape[:2]
    canvas = np.zeros((P * h, Q * w, 3), dtype=np.uint8)

    for r in range(P):
        for c in range(Q):
            if (r, c) in grid:
                piece_idx, orientation = grid[(r, c)]
                img_rotated = rotate_cv2_image(pieces[piece_idx], orientation).copy()
                img_rotated = cv2.resize(img_rotated, (w, h), interpolation=cv2.INTER_LINEAR)

                if highlight and gt_pos and gt_rot:
                    true_r, true_c = gt_pos[piece_idx]
                    true_rot = gt_rot[piece_idx]

                    if (r, c) != (true_r, true_c):
                        # Red border: Incorrect absolute placement
                        cv2.rectangle(img_rotated, (0, 0), (w - 1, h - 1), (0, 0, 255), max(1, w // 10))
                    elif orientation != true_rot:
                        # Yellow border: Correct placement, wrong rotation
                        cv2.rectangle(img_rotated, (0, 0), (w - 1, h - 1), (0, 255, 255), max(1, w // 10))

                canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img_rotated

    return canvas


def convert_to_bytes(cv2_image, target_size=(600, 600)):
    """
    Converts an OpenCV image to bytes for display in PySimpleGUI.
    :param cv2_image: The input OpenCV image.
    :param target_size: The target size for the resized image.
    :return: The image bytes.
    """
    resized = cv2.resize(cv2_image, target_size, interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.png', resized)
    return buffer.tobytes()


def main_gui():
    """
    Main function to run the GUI for the jigsaw puzzle reconstruction demo.
    """
    sg.theme('DarkBlue3')

    images = ["dog.jpg", "ocean.jpg", "acropolis.jpg"]
    grid_sizes = ["4x4", "8x8", "16x16"]
    configs = {
        "Classical Only": {'color': 0.7, 'texture': 0.3, 'cnn': 0.0, 'cnn_tile': 0.0, 'w_side': 1.0, 'w_tile': 0.0},
        "Deep Only": {'color': 0.0, 'texture': 0.0, 'cnn': 1.0, 'cnn_tile': 1.0, 'w_side': 0.7, 'w_tile': 0.3},
        "Combined": {'color': 0.4, 'texture': 0.2, 'cnn': 0.4, 'cnn_tile': 1.0, 'w_side': 0.9, 'w_tile': 0.1}
    }

    # GUI Layout
    left_column = [
        [sg.Text("Image Source:")],
        [sg.Combo(images, default_value=images[0], key="-IMAGE-", size=(20, 1))],
        [sg.Text("Grid Size:")],
        [sg.Combo(grid_sizes, default_value="4x4", key="-SIZE-", size=(20, 1))],
        [sg.Text("Feature Configuration:")],
        [sg.Combo(list(configs.keys()), default_value="Classical Only", key="-CONFIG-", size=(20, 1))],
        [sg.Button("Run Reconstruction", size=(20, 2), button_color=('white', 'green'))],
        [sg.Multiline(size=(40, 20), key="-LOG-", disabled=True, autoscroll=True)],
        [sg.Text("Diagnostic Legend:")],
        [sg.Text("■ Wrong Position", text_color="red")],
        [sg.Text("■ Wrong Rotation", text_color="yellow")]
    ]

    right_column = [
        [sg.Text("Puzzle Visualization", justification="center", font=("Helvetica", 14))],
        [sg.Image(key="-IMAGE_DISPLAY-", size=(600, 600), background_color="black")]
    ]

    layout = [[sg.Column(left_column), sg.VSeparator(), sg.Column(right_column)]]
    window = sg.Window("Jigsaw Puzzle Reconstruction Demo", layout, finalize=True)

    def log(msg):
        window["-LOG-"].print(msg)
        window.refresh()

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "Run Reconstruction":
            image_path = values["-IMAGE-"]
            config_name = values["-CONFIG-"]

            # Parse grid size
            size_str = values["-SIZE-"]
            P, Q = map(int, size_str.split('x'))
            weights = configs[config_name]

            window["-LOG-"].update("")
            log(f"Starting {P}x{Q} puzzle for {image_path}")

            # 1. Generation
            log("1. Generating and shuffling puzzle...")
            generator = PuzzleGeneration(P, Q)
            pieces, h, w = generator.generate_puzzle(image_path)

            shuffled_grid = {(r, c): (r * Q + c, 0) for r in range(P) for c in range(Q)}
            img_bytes = convert_to_bytes(render_grid_image(P, Q, pieces, shuffled_grid))
            window["-IMAGE_DISPLAY-"].update(data=img_bytes)
            window.refresh()

            # 2. Extraction
            log("2. Extracting CNN and Classical features...")
            feat_start = time.time()
            extractor = FeatureExtractor(wb=4)
            piece_features = {}
            for idx, tile in enumerate(pieces):
                strips = extractor.get_border_strips(tile)
                piece_features[idx] = {
                    'side': {
                        'cnn': {s: extractor.compute_deep_features(strip) for s, strip in strips.items()},
                        'color': {s: extractor.compute_color_histogram(strip) for s, strip in strips.items()},
                        'texture': {s: extractor.compute_texture_features(strip) for s, strip in strips.items()}
                    },
                    'tile': {
                        'cnn': extractor.compute_deep_features(tile),
                        'color': extractor.compute_color_histogram(tile)
                    }
                }
            feat_time = time.time() - feat_start
            log(f"   -> Features extracted in {feat_time:.2f}s")

            # 3. Reconstruction
            recon_start = time.time()
            log(f"3. Reconstructing using {config_name} features...")
            reconstructor = Reconstructor(P, Q, pieces, piece_features, weights, border_width=4)

            log("   -> Running greedy initialization...")
            greedy_grid = reconstructor.greedy_reconstruction()
            img_bytes = convert_to_bytes(render_grid_image(P, Q, pieces, greedy_grid))
            window["-IMAGE_DISPLAY-"].update(data=img_bytes)
            window.refresh()

            log("   -> Running simulated annealing optimization...")
            reconstructed_grid = reconstructor.optimize_grid(greedy_grid, iterations=1000)

            log("   -> Aligning global orientation...")
            evaluator = Evaluator(generator.ground_truth_positions, generator.ground_truth_rotations, P, Q)
            final_grid = reconstructor.align_global_orientation(reconstructed_grid, evaluator)

            recon_time = time.time() - recon_start
            log(f"   -> Reconstruction finished in {recon_time:.2f}s")

            # 4. Evaluation and Final Visualization
            metrics = evaluator.evaluate(final_grid)
            neigh_acc = evaluator.evaluate_neighbor_accuracy(final_grid)

            log("\n--- Final Metrics ---")
            log(f"Placement Accuracy: {metrics['placement_accuracy'] * 100:.1f}%")
            log(f"Rotation Accuracy:  {metrics['rotation_accuracy'] * 100:.1f}%")
            log(f"Neighbor Accuracy:  {neigh_acc * 100:.1f}%")
            log(f"Total Time Taken:   {(feat_time + recon_time):.2f}s")

            log("\nRendering diagnostic highlights...")
            img_bytes = convert_to_bytes(
                render_grid_image(
                    P, Q, pieces, final_grid,
                    gt_pos=generator.ground_truth_positions,
                    gt_rot=generator.ground_truth_rotations,
                    highlight=True
                )
            )
            window["-IMAGE_DISPLAY-"].update(data=img_bytes)

    window.close()


if __name__ == "__main__":
    main_gui()