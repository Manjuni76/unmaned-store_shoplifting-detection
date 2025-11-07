# Model Implementations

## STG-NF (Spatio-Temporal Graph Normalizing Flow)

### Overview
This directory contains the model implementations for the shoplifting detection system.

### Files

- `stg_nf.py`: STG-NF model implementation for pose-based anomaly detection
- `stgcn.py`: ST-GCN (Spatio-Temporal Graph Convolutional Network) - placeholder
- `feature_train.py`: Feature training utilities - placeholder
- `detection_model.py`: Detection model utilities - placeholder

### STG-NF Model

The `STG_NF` class implements a simplified version of the Spatio-Temporal Graph Normalizing Flow for pose anomaly detection.

**Key Features:**
- Encoder: ST-GCN layers to extract spatio-temporal features
- Latent representation: Fixed-size feature vector (256-dim)
- Loss: Negative log-likelihood based on standard normal prior

**Usage:**
```python
from stg_nf import STG_NF

# Initialize model
model = STG_NF(
    input_size=(2, 12, 6),  # (channels, temporal, joints)
    K=3,                     # Spatial partitions
    L=4,                     # Flow layers
    hidden_channels=512,
    device='cuda'
)

# Forward pass
z, log_det = model(x)  # x: (B, C, T, V)
loss = model.loss(z, log_det)
```

### Notes

This is a simplified implementation. For production use, consider:
- Adding proper ST-GCN layers with graph convolutions
- Implementing actual normalizing flow transformations
- Adding more sophisticated graph partitioning strategies
- Implementing proper adjacency matrix handling
