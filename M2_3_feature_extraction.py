import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class FeatureExtractor:
    def __init__(self, wb=8):
        self.wb = wb
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

        self.cnn_features = torch.nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        ).to(self.device)

        self.cnn_features.eval()

        # Use the preprocessing associated with the pretrained weights.
        self.preprocess = weights.transforms()

    def get_border_strips(self, tile):

        h, w, _ = tile.shape
        strips = {
            'top': tile[0:self.wb, :],
            'bottom': tile[h - self.wb:h, :],
            'left': tile[:, 0:self.wb],
            'right': tile[:, w - self.wb:w]
        }
        return strips

    def compute_color_histogram(self, strip, bins=8):
        hist = []
        for i in range(3):  # For each RGB channel
            channel_hist = cv2.calcHist([strip], [i], None, [bins], [0, 256])
            hist.append(channel_hist.flatten())

        hist = np.concatenate(hist)
        return hist / (np.sum(hist) + 1e-7)  # Normalize

    @torch.no_grad()
    def compute_deep_features(self, strip):
        """
        Extract a 512-dimensional ResNet18 descriptor.

        Convert the OpenCV NumPy array to a PIL Image and pass it
        through the preprocessing pipeline before feeding it to the CNN.
        """
        from PIL import Image

        strip_rgb = cv2.cvtColor(
            strip,
            cv2.COLOR_BGR2RGB
        )

        strip_pil = Image.fromarray(strip_rgb)

        input_tensor = self.preprocess(strip_pil)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        features = self.cnn_features(input_tensor)

        return features.flatten(1).cpu().numpy()[0]

    def compute_texture_features(self, strip):
        """
        Computes texture statistics using Sobel filters to capture edges.
        """
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)

        # 1st order derivatives (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        features = []
        for response in [sobelx, sobely]:
            mean = np.mean(response)
            std = np.std(response)
            energy = np.mean(response ** 2)
            features.extend([mean, std, energy])

        # Normalize the texture feature vector
        feat_array = np.array(features)
        return feat_array / (np.linalg.norm(feat_array) + 1e-7)