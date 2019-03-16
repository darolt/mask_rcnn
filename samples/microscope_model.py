"""Captures the type of microscope used in each image."""
import numpy as np
from sklearn.mixture import GaussianMixture


class MicroscopeModel:
    def __init__(self, nb_clusters):
        self.nb_clusters = nb_clusters

    def predict(self, img):
        pass

    def fit(self, imgs):
        means = self._compute_means(imgs)
        self.mixture = GaussianMixture(self.nb_clusters)
        self.mixture.fit(means)

    def _compute_means(self, imgs):
        means = [np.mean(img, axis=(0, 1)) for img in imgs]
        return np.array(means)
