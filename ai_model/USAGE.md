# Training Pipeline Usage Guide

This guide explains how to use the STG-NF training pipeline for shoplifting detection.

## Prerequisites

### 1. Environment Setup

The project requires Python 3.11 with the following dependencies:
- PyTorch (with CUDA support recommended)
- NumPy
- scikit-learn
- tqdm
- opencv-python (for data preprocessing)

Install dependencies:
```bash
conda env create -f ai_model/environment/environment.yml
conda activate unmaned_shoplifting
```

Or with pip:
```bash
pip install torch torchvision numpy scikit-learn tqdm opencv-python
```

### 2. Data Preparation

The pipeline expects data in the following structure:

```
data_split/output/
├── train_data.json          # Normal data for STG-NF training
├── mlp_train_data.json      # Normal + Abnormal for MLP training
└── test_data.json           # Test data

data/
├── train_data_skeleton_data/
├── mlp_train_data_skeleton_data/
└── test_data_skeleton_data/
```

Each JSON file should contain:
```json
{
  "normal": [
    {
      "filename": "video1.mp4",
      "full_path": "/path/to/video1.mp4",
      "label": 0
    }
  ],
  "abnormal": [
    {
      "filename": "video2.mp4",
      "full_path": "/path/to/video2.mp4",
      "label": 1,
      "theft_start": 100,
      "theft_end": 200,
      "total_frames": 300
    }
  ]
}
```

Skeleton data should be JSON files in the format:
```json
[
  {
    "keypoints": [x1, y1, c1, x2, y2, c2, ..., x18, y18, c18]
  }
]
```

## Quick Start

### Option 1: Using the Test Script (Recommended for First-Time Setup)

```bash
cd ai_model
python test_model.py
```

This will validate that:
- STG-NF model can be imported
- Model can be instantiated
- Forward pass works correctly
- Training pipeline imports are functional

### Option 2: Running the Full Training Pipeline

**Important:** Before running, update the data paths in `train_pipeline.py`:

```python
args = {
    # Update these paths to match your data location
    'train_json': '../data_split/output/train_data.json',
    'mlp_train_json': '../data_split/output/mlp_train_data.json',
    'test_json': '../data_split/output/test_data.json',
    
    'train_skeleton_path': '../data/train_data_skeleton_data',
    'mlp_skeleton_path': '../data/mlp_train_data_skeleton_data',
    'test_skeleton_path': '../data/test_data_skeleton_data',
    
    # ... other parameters
}
```

Then run:
```bash
cd ai_model
python train_pipeline.py
```

## Training Process

The pipeline follows three stages:

### Stage 1: STG-NF Training (Normal Data Only)
- **Data**: Normal behavior patterns (train_data.json)
- **Purpose**: Learn the distribution of normal poses
- **Output**: `checkpoints/stgnf_arms.pth`
- **Duration**: ~50 epochs (configurable)

### Stage 2: MLP Classifier Training
- **Data**: Both normal and abnormal data (mlp_train_data.json)
- **Purpose**: Learn to classify normal vs abnormal using STG-NF features
- **STG-NF**: Frozen (no training)
- **MLP**: Trainable
- **Output**: `checkpoints/full_model_arms.pth`
- **Duration**: ~30 epochs (configurable)

### Stage 3: Evaluation
- **Data**: Test set (test_data.json)
- **Metrics**: Accuracy, AUC-ROC, AUC-PR

## Configuration

### Body Part Selection

You can train on different body parts:

```python
from train_pipeline import JOINT_SUBSET_MAP

args['joint_subset'] = JOINT_SUBSET_MAP['arms']      # Arms only (default)
args['joint_subset'] = JOINT_SUBSET_MAP['legs']      # Legs only
args['joint_subset'] = JOINT_SUBSET_MAP['body']      # Torso
args['joint_subset'] = JOINT_SUBSET_MAP['all']       # All joints (None)
```

### Hyperparameters

```python
args = {
    'seg_len': 12,           # Sequence length (frames)
    'seg_stride': 6,         # Stride for sliding window
    'batch_size': 32,        # Batch size
    'epochs_stgnf': 50,      # STG-NF training epochs
    'epochs_mlp': 30,        # MLP training epochs
    'lr_stgnf': 1e-4,        # STG-NF learning rate
    'lr_mlp': 1e-3,          # MLP learning rate
    'seed': 42               # Random seed
}
```

## Troubleshooting

### Issue: "No module named 'numpy'"
**Solution**: Install dependencies (see Prerequisites)

### Issue: "FileNotFoundError: train_data.json"
**Solution**: Update data paths in `train_pipeline.py` or run `data_split.py` first

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size:
```python
args['batch_size'] = 16  # or 8
```

### Issue: "No skeleton files found"
**Solution**: Extract skeleton data first using `extract_skeleton/` scripts

## Expected Output

```
================================================================================
STEP 1: STG-NF 정상 데이터 학습
================================================================================
[DATASET] 총 783개 영상 로드
[DATASET] 총 15660개 세그먼트 생성
Epoch 1/50, Loss: 2.3456
Epoch 10/50, Loss: 1.5432
...
[SAVE] STG-NF 모델 저장: checkpoints/stgnf_arms.pth

================================================================================
STEP 2: MLP 분류기 학습 (정상+이상 데이터)
================================================================================
[DATASET] 총 712개 영상 로드
Epoch 1/30, Loss: 0.6543, Acc: 65.43%
...
[SAVE] Full 모델 저장: checkpoints/full_model_arms.pth

================================================================================
STEP 3: Test 데이터 평가
================================================================================
[RESULTS]
Accuracy: 87.34%
AUC-ROC: 0.9234
AUC-PR: 0.8976
```

## Next Steps

After successful training:

1. **Evaluate on different body parts**: Train separate models for legs, body, head
2. **Ensemble models**: Combine predictions from multiple body part models
3. **Fine-tune hyperparameters**: Experiment with learning rates, epochs, etc.
4. **Deploy the model**: Use the trained model for real-time detection

## Additional Resources

- Model documentation: `model/README.md`
- Data preprocessing: `data_preprocessing/`
- Main project README: `../README.md`
