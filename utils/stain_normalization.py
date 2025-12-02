def macenko_normalization(img):
    return img   # placeholder
import numpy as np
import cv2
from typing import Optional, Tuple

class MacenkoStainNormalizer:
    """Macenko stain normalization for H&E images."""
    
    def __init__(self):
        self.target_stains = None
        self.target_concentrations = None
        self.maxC_target = None
        
    def fit(self, target: np.ndarray):
        """Fit normalizer to target image."""
        self.target_stains = self._get_stain_matrix(target)
        self.target_concentrations = self._get_concentrations(target, self.target_stains)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=1, keepdims=True)
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to target distribution."""
        if self.target_stains is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        # Get stain matrix and concentrations for source image
        source_stains = self._get_stain_matrix(image)
        source_concentrations = self._get_concentrations(image, source_stains)
        maxC_source = np.percentile(source_concentrations, 99, axis=1, keepdims=True)
        
        # Normalize concentrations
        source_concentrations *= (self.maxC_target / maxC_source)
        
        # Recreate image
        normalized = 255 * np.exp(-self.target_stains @ source_concentrations)
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        normalized = normalized.T.reshape(image.shape)
        
        return normalized
    
    def _get_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        """Extract stain matrix using method of Macenko et al."""
        # Convert to OD space
        od = self._rgb_to_od(image)
        od = od.reshape((-1, 3))
        
        # Remove background (low OD)
        od_hat = od[~np.any(od < 0.15, axis=1)]
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
        eigvecs = eigvecs[:, [2, 1]]  # Take top 2
        
        # Project and find extreme angles
        proj = od_hat @ eigvecs
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        
        # Find angles corresponding to H and E
        min_angle = np.percentile(angles, 1)
        max_angle = np.percentile(angles, 99)
        
        # Convert back to OD space
        h = eigvecs @ np.array([np.cos(min_angle), np.sin(min_angle)])
        e = eigvecs @ np.array([np.cos(max_angle), np.sin(max_angle)])
        
        # Normalize
        h = h / np.linalg.norm(h)
        e = e / np.linalg.norm(e)
        
        return np.array([h, e])
    
    def _get_concentrations(self, image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        """Get concentration matrix."""
        od = self._rgb_to_od(image).reshape((-1, 3))
        concentrations = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0]
        return concentrations
    
    @staticmethod
    def _rgb_to_od(image: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density."""
        image = image.astype(np.float32) + 1
        return -np.log(image / 255)


class ReinhardStainNormalizer:
    """Reinhard color normalization."""
    
    def __init__(self):
        self.target_means = None
        self.target_stds = None
    
    def fit(self, target: np.ndarray):
        """Fit to target image."""
        lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
        self.target_means, self.target_stds = self._get_mean_std(lab)
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform image."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        source_means, source_stds = self._get_mean_std(lab)
        
        # Normalize
        lab = lab.astype(np.float32)
        for i in range(3):
            lab[:, :, i] = ((lab[:, :, i] - source_means[i]) * 
                           (self.target_stds[i] / source_stds[i]) + 
                           self.target_means[i])
        
        lab = np.clip(lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    @staticmethod
    def _get_mean_std(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and std per channel."""
        means = np.mean(image, axis=(0, 1))
        stds = np.std(image, axis=(0, 1))
        return means, stds