# RetinexFormer-Enhanced Mamba-ViT for Pulmonary Tuberculosis Detection

A hybrid deep learning framework combining RetinexFormer image enhancement with Mamba-ViT dual-encoder architecture for automated tuberculosis classification and lung segmentation from chest X-rays.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

This repository implements a multi-task deep learning model that simultaneously performs:
- **Binary Classification**: TB-positive vs. Normal chest radiographs
- **Lung Segmentation**: Precise delineation of lung fields

### Key Features

- **RetinexFormer Enhancement**: Learnable illumination normalization for robust preprocessing across varying image quality
- **Dual-Encoder Architecture**: Mamba-style CNN for local features + Vision Transformer for global context
- **Cross-Attention Fusion**: Effective integration of complementary feature representations
- **Uncertainty-Weighted Loss**: Automatic task balancing without manual hyperparameter tuning
- **Comprehensive Visualization**: Training curves, GradCAM, predictions, and evaluation metrics

---

## Performance

| Task | Metric | Value |
|------|--------|-------|
| Classification | Accuracy | 99.64% |
| Classification | AUC-ROC | 0.9999 |
| Classification | Sensitivity | 97.86% |
| Classification | Specificity | 100% |
| Segmentation | Dice Coefficient | 0.962 |
| Segmentation | IoU | 0.928 |

---

## Installation

```bash
# Clone repository
git clone https://github.com/Maisamilens/tb-detection-mamba-vit.git
cd tb-detection-mamba-vit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
Pillow>=9.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
scipy>=1.7.0
```

---

## Dataset Structure

```
project_root/
├── classification/
│   └── TB_Chest_Radiography_Database/
│       ├── Normal/
│       │   └── *.png
│       └── Tuberculosis/
│           └── *.png
├── segmentation/
│   └── Lung Segmentation/
│       ├── CXR_png/
│       │   └── *.png
│       └── masks/
│           └── *_mask.png
└── tb_detection.py
```

### Supported Datasets

- **Classification**: [NIAID TB Portal](https://tbportals.niaid.nih.gov/) / Kaggle TB Chest X-ray Dataset
- **Segmentation**: Montgomery County + Shenzhen Hospital datasets
- Dataset for this paper is available at (https://drive.google.com/drive/folders/1BhSnRUF6x98MIbt_TfWc6QgRCPcj7s2h?usp=sharing)

---

## Usage

### Training

```bash
python tb_detection.py
```

Training parameters can be modified in the `CONFIG` dictionary:

```python
CONFIG = {
    'img_size': (512, 512),
    'batch_size': 4,
    'epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'patience': 15,        # Early stopping
    'grad_clip': 1.0,
    'seed': 42
}
```

### Inference

```python
import torch
from tb_detection import TBHybridModel

# Load model
model = TBHybridModel(img_size=512).to('cuda')
checkpoint = torch.load('results/checkpoints/best_auc_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    cls_prob = torch.sigmoid(outputs['cls_logits']).item()
    seg_mask = outputs['seg_probs']
```

---

## Model Architecture

```
Input Image (512×512)
       │
       ▼
┌─────────────────┐
│  RetinexFormer  │ ──► Illumination Normalization
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Mamba │ │  ViT  │
│Encoder│ │Encoder│
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│ Cross-Attention │
│     Fusion      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│  Cls  │ │  Seg  │
│ Head  │ │Decoder│
└───┬───┘ └───┬───┘
    ▼         ▼
 TB/Normal  Lung Mask
```

---

## Output Structure

```
results/
├── checkpoints/
│   ├── best_auc_model.pth
│   ├── best_dice_model.pth
│   └── final_model.pth
├── visualizations/
│   ├── training_progress_epoch_*.png
│   └── predictions_epoch_*.png
├── metrics/
│   ├── training_history.json
│   ├── validation_report.json
│   ├── *_classification_curves.png
│   └── *_segmentation_distribution.png
└── logs/
    └── training.log
```

---

## Results Visualization

The framework generates comprehensive visualizations including:

- **Training Curves**: Loss, AUC, Dice, F1, Precision/Recall, Learning Rate
- **ROC & PR Curves**: With optimal threshold selection
- **Confusion Matrix**: Classification performance breakdown
- **Segmentation Examples**: Best, median, and worst cases with Dice scores
- **Prediction Overlays**: Side-by-side comparison of predictions and ground truth

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tbdetection2025,
  title={RetinexFormer-Enhanced Mamba-ViT Hybrid Model for Pulmonary Tuberculosis Classification and Segmentation in Public Health Screening},
  author={Maisam Abbas, Anam Munir, Ran-Zan Wang},
  journal={Under Progress},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [NIAID TB Portal](https://tbportals.niaid.nih.gov/) for tuberculosis imaging data
- Montgomery County and Shenzhen Hospital for lung segmentation datasets
- PyTorch team for the deep learning framework

---

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
