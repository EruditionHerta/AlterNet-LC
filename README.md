# AlterNet-LC: Pneumonia Detection Model

This repository contains the code and description for **AlterNet-LC**, a deep learning model designed for Pneumonia detection using chest X-ray images. The primary codebase is provided within a Jupyter Notebook (`Code & Description.ipynb`).

The notebook includes:

  * Complete training code for AlterNet-LC, encompassing data preprocessing, model definition, training, validation, and testing procedures.

**Authors:** Li Jiawei, Chen Mingfang, Yao Zehan

**⚠️ Warning:**

> The code in this notebook is for reference only. It has **not undergone strict data leakage prevention or logic optimization** and should **not be directly used in production environments**.
> Results may vary due to device differences. Default configurations are provided for reference; adjust and debug according to your actual setup.

## Datasets Used

  * **Primary Training/Validation/Testing Dataset:** PneumoniaMNIST (part of MedMNIST).
      * The notebook expects this dataset to be available as `pneumoniamnist_224.npz`.
  * **Generalization Testing Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle (`paultimothymooney/chest-xray-pneumonia`).

## Setup and Installation

1.  **Clone the repository (or download the notebook).**
2.  **Install dependencies:**
      * **MedMNIST:** For accessing the PneumoniaMNIST dataset.
        ```bash
        pip install medmnist
        ```
      * **Kaggle Hub:** For downloading the Kaggle dataset.
        ```bash
        pip install kagglehub
        ```
      * **Core Libraries:** The notebook uses `numpy`, `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, and `tqdm`. Install them as needed (e.g., via `pip` or `conda`).
        ```bash
        pip install numpy torch torchvision scikit-learn matplotlib seaborn tqdm
        ```
3.  **Download Datasets:**
      * **PneumoniaMNIST:**
        The notebook can download this dataset if it's not found locally, using the `medmnist` library. The downloaded data will be processed into `pneumoniamnist_224.npz` or used directly.

        ```python
        # Example snippet from the notebook for downloading
        import medmnist
        from medmnist import INFO
        import torchvision.transforms as transforms

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        data_flag = 'pneumoniamnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])

        # This will download the data if not present
        train_dataset = DataClass(split='train', transform=data_transform, download=True, size=224, mmap_mode='r')
        ```

        *Note: Downloading MedMNIST datasets may require a VPN connection in some regions.*
        The notebook is configured to load data from `pneumoniamnist_224.npz`.

      * **Kaggle Chest X-Ray Dataset (for generalization testing):**
        The notebook provides instructions to download this using `kagglehub`.

        ```python
        import kagglehub

        # May require login for the first time
        # kagglehub.login()

        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Path to dataset files:", path)
        ```

## AlterNet-LC Model Architecture

AlterNet-LC is a hybrid model integrating convolutional neural networks (CNNs) with self-attention mechanisms. Key components include:

  * **Stem:** Initial convolutional layers (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.MaxPool2d`) for low-level feature extraction.
  * **Pre-activated Residual Blocks (`PreActResidualBlock`):** Used for building deeper CNN stages, featuring batch normalization and ReLU activation before convolution.
  * **WindowAttention:** A utility to partition feature maps into non-overlapping windows for localized self-attention, and to reverse this process.
  * **Multiple Self-attention Blocks (`MSABlock`):** These blocks perform self-attention within windows. They include:
      * Layer normalization (`nn.LayerNorm`).
      * Multi-head self-attention (QKV computation, scaled dot-product attention).
      * MLP layers with GELU activation and dropout.
      * Residual connections.
  * **Contrastive Self-attention Blocks (`ContrastiveMSABlock`):** An extension of `MSABlock` that incorporates a contrastive learning head.
      * A projection head maps features to a space for contrastive learning.
      * Calculates a contrastive loss based on feature similarity and class labels during training.
  * **Hybrid Stages:** The model architecture consists of an initial pure CNN stage, followed by stages that alternate `PreActResidualBlock`s with `ContrastiveMSABlock`s. This allows the model to learn both local and global features, enhanced by contrastive learning.
  * **Classification Head:** Global average pooling (`nn.AdaptiveAvgPool2d`), dropout, and a final fully connected layer (`nn.Linear`) for classification.

## Loss Function: LMF Loss

The model is trained using a custom loss function named **LMFLoss**. This loss combines:

  * **Focal Loss:** To address class imbalance by down-weighting the loss assigned to well-classified examples.
  * **Margin Loss:** To enforce a margin between class probabilities, enhancing separability.
    The LMFLoss also incorporates class weights and specific parameters (`gamma`, `margin`, `alpha`) to fine-tune its behavior. An additional weighting is applied for misclassified negative samples.

## Core Functionality (in `Code & Description.ipynb`)

The Jupyter Notebook is structured to provide a comprehensive workflow:

1.  **Data Loading and Preprocessing (`load_and_preprocess_data`):**
      * Loads data from the `pneumoniamnist_224.npz` file.
      * Ensures images are 3-channel (RGB).
      * Applies data augmentation to the training set (RandomHorizontalFlip, RandomRotation, ColorJitter).
      * Normalizes images and converts them to PyTorch tensors.
      * Creates `DataLoader` instances for training, validation, and testing.
2.  **Custom Dataset Class (`MedicalDataset`):**
      * A `torch.utils.data.Dataset` subclass to handle medical images and their labels, applying transformations.
3.  **Model Training (`train_model`):**
      * Implements the training and validation loop.
      * Uses the **AdamW optimizer** and **ReduceLROnPlateau learning rate scheduler**.
      * Calculates the combined loss: `LMF Loss + contrastive_weight * contrastive_loss`.
      * Tracks and logs training/validation metrics (loss, accuracy, AUC).
      * Implements **early stopping** based on validation loss.
      * Saves the best performing model state during training.
4.  **Model Evaluation (`evaluate_model`):**
      * Assesses the trained model on the test set.
      * Calculates a comprehensive set of metrics: accuracy, precision, recall, specificity, F1-score, Negative Predictive Value (NPV), ROC AUC, and PR AUC (Average Precision).
      * Generates and displays a confusion matrix and classification report.
      * Saves detailed evaluation results to a `.txt` file in the `evaluation_results` directory.
5.  **Visualization:**
      * `plot_training`: Plots training and validation accuracy and loss curves over epochs.
      * The notebook also includes code to plot ROC and PR curves for the test set results.

## How to Run

1.  Ensure all prerequisites (dependencies, datasets) are met as described in the "Setup and Installation" section.
2.  The primary dataset file `pneumoniamnist_224.npz` should be in the location specified by `data_path` in the notebook (defaults to the same directory).
3.  Open and run the cells in the `Code & Description.ipynb` Jupyter Notebook.
4.  Key sections to execute:
      * **Data Loading:** The cell calling `load_and_preprocess_data`.
      * **Model Initialization:** The cell defining and instantiating `AlterNet_LC`.
      * **Training:** The cell calling `train_model`. This will train the model and save the best version (e.g., `alternet_contrastive_YYYYMMDD.pth`).
      * **Plotting Training History:** The cell calling `plot_training`.
      * **Evaluation:** The cell calling `evaluate_model`. This will print metrics and save results to a text file.
      * **Plotting Evaluation Curves:** Cells for plotting ROC and PR curves.
      * **Saving Final Model:** The final cell saves the model state (e.g., `alternet_lc_YYYYMMDD.pth`).

## Expected Outputs

  * **Trained Model Files:**
      * `alternet_contrastive_YYYYMMDD.pth` (best model saved during training based on validation loss).
      * `alternet_lc_YYYYMMDD.pth` (final model saved at the end of the notebook).
  * **Evaluation Results Directory (`evaluation_results/`):**
      * A text file (e.g., `evaluation_results_YYYYMMDD_HHMMSS.txt`) containing detailed performance metrics, confusion matrix, and classification report.
      * An `images/` subdirectory is also created, though the notebook doesn't explicitly save plots there via `plt.savefig()`, it is prepared for such use.
  * **Plots:**
      * Training/Validation accuracy and loss curves.
      * ROC curve with AUC.
      * Precision-Recall (PR) curve with Average Precision (AP).
        (These are displayed within the notebook).

## Repository Structure (derived from the notebook)

```
.
├── Code & Description.ipynb    # Main Jupyter Notebook with all code and explanations
```
