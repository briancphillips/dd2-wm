# DynaDetect-WM Implementation Plan

## Phase 1: Environment & Data Pipeline (Steps 1 & 2)
- **Initialize repository**: Set up the Git tracking, Python virtual environment, and install dependencies (`torch`, `torchvision`, `scikit-learn`, etc.) compatible with CUDA 12.8.
- **Dataset Loaders**: 
  - Write automated download and preprocessing scripts for CIFAR-100 and GTSRB.
  - Set up local paths and custom `torch.utils.data.Dataset` classes for VGGFace2 and CheXpert (which typically require manual credentialed downloads).
- **Architecture Base**: Implement standard CNN architectures (e.g., ResNet, DenseNet) for baseline training to ensure the data pipeline is functional.

## Phase 2: Attack Simulation & Detection (Steps 3 & 4)
- **Gradient Manipulation Attack**: Implement the gradient-based poisoning mechanism (e.g., referencing "Witches' Brew" from the PDF). 
  - Create a script to generate poisoned variants of 5-10% of the training data.
  - Verify the attack succeeds on a baseline model.
- **DynaDetect2.0 Integration**: 
  - Port over your existing `DynaDetect2.0` logic.
  - Implement the CNN feature extractor and the Mahalanobis distance-based anomaly scorer.
  - Run the detector over the poisoned dataset and capture the flagged indices.

## Phase 3: DynaDetect-WM Watermarking & Tracing Module (Steps 5, 7, & 8)
- **Watermark Repurposing**: Create a module that takes the flagged samples from Phase 2 and isolates them as "Watermark Probes".
- **Tracking Architecture**: Develop the `WatermarkMonitor` system.
  - Use PyTorch forward hooks to capture intermediate layer activations.
  - Define the algorithm for calculating latent drift and activation distance metrics between clean models and models exposed to the watermarked data.
- **Intentional Embedding**: Mix the watermark probes back into a clean training set to simulate releasing the protected dataset.

## Phase 4: Model Auditing & Evaluation (Step 6)
- **Unauthorized Training Simulation**: Train "stolen" target models using the watermarked dataset.
- **Audit Execution**: Run the `WatermarkMonitor` against the stolen model using the watermark probes.
- **Metrics Collection**: Output confidence scores, activation alignment, and classification behavior to conclusively prove the model trained on your proprietary data.

## Phase 5: Documentation & Literature (Step 9)
- Compile tracking metrics into visualizations (matplotlib/seaborn).
- Maintain a running bibliography of papers focusing on gradient-based poisoning and latent space anomaly detection.
