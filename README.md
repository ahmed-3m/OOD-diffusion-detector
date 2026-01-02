# OOD-Diffusion-Detector

A binary diffusion-based classifier for out-of-distribution (OOD) detection. This project demonstrates how a diffusion model trained on a single class (airplanes) can be used to detect out-of-distribution samples.

## Overview

This project implements and evaluates a novel approach to OOD detection using diffusion models. The key idea is to:

1. Train a diffusion model on a single class (CIFAR-10 airplanes)
2. Use the trained model to measure the noise prediction error on samples
3. Classify samples as in-distribution (airplane) or out-of-distribution (not airplane) based on prediction error

The project also compares this approach to a traditional discriminative classifier (ResNet20 trained on CIFAR-10).

## Project Structure

```
OOD-diffusion-detector/
├── checkpoints/           # Saved model checkpoints
├── configs/               # Configuration files
├── data/                  # Dataset storage (created on first run)
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks for exploration and visualization
├── scripts/               # Training and evaluation scripts
│   ├── evaluate.py        # Script to evaluate trained models
│   └── train.py           # Script to train the diffusion model
├── src/                   # Source code
│   ├── datasets/          # Dataset classes
│   │   ├── cifar_dataset.py     # Test dataset for CIFAR-10
│   │   └── data_module.py       # PyTorch Lightning DataModule
│   ├── evaluation/        # Evaluation utilities
│   │   └── classifier.py        # Classification and evaluation functions
│   ├── models/            # Model definitions
│   │   └── diffusion_model.py   # Diffusion model implementation
│   └── visualization/     # Visualization utilities
├── utils/                 # Utility functions
├── .gitignore             # Git ignore file
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── setup.py               # Package installation script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ahmed-3m/OOD-diffusion-detector.git
cd OOD-diffusion-detector
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the diffusion model:

```bash
python scripts/train.py --output_dir ./checkpoints --experiment_name "diffusion_training"
```

Options:
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 3501)
- `--devices`: Number of GPUs to use (default: 1)
- `--offline`: Run without WandB logging
- `--checkpoint_path`: Resume training from a checkpoint
- `--precision`: Training precision (default: "bf16-mixed")

### Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint_path ./checkpoints/last.ckpt
```

Options:
- `--num_samples`: Number of samples to evaluate (default: 1000)
- `--offline`: Run without WandB logging
- `--experiment_name`: Name for the evaluation experiment

## Results

The evaluation compares the diffusion-based classifier with a traditional discriminative classifier on the binary task of distinguishing airplanes from other CIFAR-10 classes. Results include:

- Accuracy and AUROC for both classifiers
- Confusion matrices
- Comparative visualizations
- Data tables for thesis/paper use

Results are logged to WandB and saved as CSV files for further analysis.

## References

This work builds on several key papers in the field:

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
2. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. CVPR.
3. Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). Variational diffusion models. NeurIPS.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code for your research, please cite:

```
@software{OODDiffusionDetector,
  author = {Ahmed Mohammed},
  title = {OOD-Diffusion-Detector: A Binary Diffusion-Based OOD Detector},
  year = {2023},
  url = {https://github.com/ahmed-3m/OOD-diffusion-detector}
}
```
