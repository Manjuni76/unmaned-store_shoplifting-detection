"""
STG-NF (Spatio-Temporal Graph Normalizing Flow) Model
Simplified implementation for pose anomaly detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STG_NF(nn.Module):
    """
    Simplified STG-NF model for pose-based anomaly detection
    """
    def __init__(self, input_size, K=3, L=4, hidden_channels=512, device='cpu', subset_idx=None):
        """
        Args:
            input_size: tuple (C, T, V) where C=channels, T=temporal, V=vertices/joints
            K: Number of spatial partition subsets
            L: Number of flow layers
            hidden_channels: Hidden dimension size
            device: Device to run model on
            subset_idx: Joint subset indices (None for all joints)
        """
        super(STG_NF, self).__init__()
        
        self.C, self.T, self.V = input_size
        self.K = K
        self.L = L
        self.hidden_channels = hidden_channels
        self.device = device
        self.subset_idx = subset_idx
        
        # Encoder: ST-GCN layers to extract features
        self.encoder = nn.Sequential(
            nn.Conv2d(self.C, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Adaptive pooling to get fixed-size representation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension after pooling
        self.feature_dim = 256
        
        # Pre-compute constant for loss computation
        self.log_2pi = np.log(2 * np.pi)
        
    def encode(self, x):
        """
        Encode input pose sequence to latent representation
        Args:
            x: (B, C, T, V) pose sequence
        Returns:
            z: (B, feature_dim) latent features
            log_det: log determinant (placeholder for compatibility)
        
        Note: This is a simplified implementation. A full normalizing flow
        would compute the actual log determinant of the Jacobian for each
        transformation layer. This placeholder returns zeros for compatibility
        with the training pipeline.
        """
        B = x.size(0)
        
        # Reshape for 2D convolution: (B, C, T, V)
        features = self.encoder(x)  # (B, 256, T, V)
        
        # Pool to fixed size
        z = self.adaptive_pool(features)  # (B, 256, 1, 1)
        z = z.view(B, -1)  # (B, 256)
        
        # Placeholder log determinant for compatibility
        # In a full NF implementation, this would be the sum of log determinants
        # from all invertible transformation layers
        log_det = torch.zeros(B, device=x.device)
        
        return z, log_det
    
    def forward(self, x):
        """
        Forward pass through the normalizing flow
        Args:
            x: (B, C, T, V) pose sequence
        Returns:
            z: latent representation
            log_det: log determinant
        """
        return self.encode(x)
    
    def loss(self, z, log_det):
        """
        Compute negative log-likelihood loss for normalizing flow
        Args:
            z: latent features (B, D)
            log_det: log determinant (B,)
        Returns:
            loss: scalar loss value
        """
        # Assume standard normal prior: p(z) = N(0, I)
        # log p(z) = -0.5 * (z^2 + log(2*pi))
        # Use pre-computed log(2*pi) constant
        log_pz = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.size(1) * self.log_2pi
        
        # log p(x) = log p(z) + log |det(dz/dx)|
        log_px = log_pz + log_det
        
        # Negative log-likelihood
        nll = -torch.mean(log_px)
        
        return nll