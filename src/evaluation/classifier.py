import torch
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

@torch.no_grad()
def classify_image(model, image, num_trials=10):
    """
    Unconditional Diffusion Classifier
    Following Algorithm 1 but adapted for unconditional case:
    1. For each trial:
       - Sample timestep t and noise ε
       - Create noisy image x_t
       - For each class:
         - Predict noise and compute error
    2. Return class with minimum mean error
    
    Args:
        model (DiffusionModel): Trained diffusion model
        image (torch.Tensor): Input image [3, 32, 32]
        num_trials (int): Number of trials T
        
    Returns:
        predicted_label (int): Class index with lowest noise prediction error
        mean_errors (np.array): Mean errors for each class
    """
    model.eval()
    device = model.device
    image = image.to(device).unsqueeze(0)
    num_classes = 2
    
    # Initialize error lists for each class (following algorithm)
    Errors = [[] for _ in range(num_classes)]
    
    # For each trial j = 1,...,T
    for _ in range(num_trials):
        # Sample t ~ [1, 1000]
        t = torch.randint(1, model.scheduler.config.num_train_timesteps, (1,), device=device)
        
        # Sample ε ~ N(0, I)
        epsilon = torch.randn_like(image)
        
        # Create x_t
        x_t = model.scheduler.add_noise(image, epsilon, t)
        
        # For each class
        x_t_batch = x_t.repeat(num_classes, 1, 1, 1)
        t_batch = t.repeat(num_classes)
        
        # Predict noise ε_θ(x_t)
        epsilon_theta = model.model(x_t_batch, t_batch).sample
        
        # Compute ||ε - ε_θ(x_t)||²
        epsilon_expanded = epsilon.repeat(num_classes, 1, 1, 1)
        error = torch.nn.functional.mse_loss(epsilon_theta, epsilon_expanded, reduction="none")
        error = error.mean(dim=[1, 2, 3])
        
        # Append errors
        for k in range(num_classes):
            Errors[k].append(error[k].item())
    
    # Convert to tensor for computation
    Errors = torch.tensor(Errors, device=device)
    
    # Return argmin mean(Errors[c_i])
    mean_errors = Errors.mean(dim=1)
    predicted_label = int(mean_errors.argmin().item())
    
    return predicted_label, mean_errors.cpu().numpy()

def get_pretrained_cifar10_model(device):
    """
    Load a pretrained ResNet20 model fine-tuned on CIFAR-10.
    
    Args:
        device: The device to load the model on
        
    Returns:
        model: Pretrained CIFAR-10 classifier
    """
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def classify_with_discriminative(model, image):
    """
    Classify a single test image using a discriminative CNN model.

    Args:
        model: Pretrained discriminative model.
        image (torch.Tensor): Tensor of shape [3, 32, 32] normalized as during training.

    Returns:
        predicted_label (int): Class index (0-1) with the highest probability.
        probs (np.array): Array of probabilities for each class.
    """
    model.eval()
    # Ensure proper normalization for the discriminative model
    # For ResNet, the normalization is different from diffusion model
    image_normalized = (image * 0.5) + 0.5  # Convert from [-1, 1] to [0, 1]

    # Apply ImageNet normalization (required for pretrained models)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_normalized)

    # Add batch dimension
    image_normalized = image_normalized.unsqueeze(0).to(next(model.parameters()).device)

    # Forward pass
    outputs = model(image_normalized)
    probs = torch.softmax(outputs, dim=1)

    # Get full CIFAR-10 prediction
    full_predicted_label = int(probs.argmax(dim=1).item())
    
    # For consistency with the diffusion model, convert to binary classification
    # where 0 = airplane, 1 = not airplane
    binary_label = 0 if full_predicted_label == 0 else 1
    binary_probs = torch.zeros(2)
    binary_probs[0] = probs[0, 0]  # Airplane probability
    binary_probs[1] = 1 - probs[0, 0]  # Not airplane probability
    
    return binary_label, binary_probs.cpu().numpy()

def evaluate_models(diffusion_model, discriminative_model, test_loader, num_trials=10, num_samples=None):
    """
    Evaluate and compare diffusion and discriminative classification models.

    Args:
        diffusion_model: Trained diffusion model.
        discriminative_model: Trained discriminative classifier.
        test_loader: DataLoader for test set.
        num_trials: Number of trials for diffusion classifier.
        num_samples: Maximum number of samples to evaluate. If None, use all samples.

    Returns:
        metrics: Dict containing comparison metrics.
    """
    device = next(diffusion_model.parameters()).device

    # For storing predictions and metrics
    true_labels = []
    diffusion_preds = []
    discriminative_preds = []

    sample_count = 0

    for images, labels in test_loader:
        for i, (image, label) in enumerate(zip(images, labels)):
            true_labels.append(label.item())

            # Diffusion classification
            diff_pred, _ = classify_image(diffusion_model, image, num_trials)
            diffusion_preds.append(diff_pred)

            # Discriminative classification
            disc_pred, _ = classify_with_discriminative(discriminative_model, image)
            discriminative_preds.append(disc_pred)

            sample_count += 1
            if num_samples is not None and sample_count >= num_samples:
                break

        if num_samples is not None and sample_count >= num_samples:
            break

    # Calculate metrics
    diff_accuracy = accuracy_score(true_labels, diffusion_preds)
    disc_accuracy = accuracy_score(true_labels, discriminative_preds)

    diff_conf_matrix = confusion_matrix(true_labels, diffusion_preds)
    disc_conf_matrix = confusion_matrix(true_labels, discriminative_preds)
    
    # Calculate AUROC if possible (requires probability scores)
    try:
        # Convert preds to binary format for ROC calculation
        auroc = roc_auc_score(true_labels, diffusion_preds)
    except (ValueError, TypeError):
        # Handle cases where ROC can't be calculated
        auroc = 0.5  # Default value for random classifier
    
    metrics = {
        'sample_count': sample_count,
        'true_labels': true_labels,
        'diffusion_preds': diffusion_preds,
        'discriminative_preds': discriminative_preds,
        'diffusion_accuracy': diff_accuracy,
        'discriminative_accuracy': disc_accuracy,
        'diffusion_conf_matrix': diff_conf_matrix,
        'discriminative_conf_matrix': disc_conf_matrix,
        'auroc': auroc  # Add AUROC to metrics
    }

    return metrics

def log_confusion_matrices(metrics, class_names=None):
    """
    Log confusion matrices to wandb.
    
    Args:
        metrics: Dict containing evaluation metrics
        class_names: Optional list of class names
    """
    try:
        # Initialize wandb if not already initialized
        if wandb.run is None:
            wandb.init(project="OOD_diffusion_detector", name="model_comparison")
        
        # Get metrics
        diff_conf_matrix = metrics['diffusion_conf_matrix']
        disc_conf_matrix = metrics['discriminative_conf_matrix']
        
        # Set class names for binary classification
        binary_class_names = ["Airplane", "Not Airplane"]
        
        # Plot diffusion confusion matrix (always binary)
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff_conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=binary_class_names, yticklabels=binary_class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Diffusion Classifier Confusion Matrix')
        wandb.log({"diffusion_confusion_matrix": wandb.Image(plt)})
        plt.close()

        # Plot discriminative confusion matrix (should also be binary after our fix)
        plt.figure(figsize=(10, 8))
        sns.heatmap(disc_conf_matrix, annot=True, fmt='d', cmap='Greens',
                    xticklabels=binary_class_names, yticklabels=binary_class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Discriminative Classifier Confusion Matrix')
        wandb.log({"discriminative_confusion_matrix": wandb.Image(plt)})
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to log confusion matrices to WandB: {e}")
        # Save plots locally as fallback
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff_conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=binary_class_names, yticklabels=binary_class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Diffusion Classifier Confusion Matrix')
        plt.savefig("diffusion_confusion_matrix.png")
        plt.close()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(disc_conf_matrix, annot=True, fmt='d', cmap='Greens',
                    xticklabels=binary_class_names, yticklabels=binary_class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Discriminative Classifier Confusion Matrix')
        plt.savefig("discriminative_confusion_matrix.png")
        plt.close()

def log_comparison_charts(metrics):
    """
    Log comparison charts to wandb.
    
    Args:
        metrics: Dict containing evaluation metrics
    """
    try:
        # Initialize wandb if not already initialized
        if wandb.run is None:
            wandb.init(project="OOD_diffusion_detector", name="model_comparison")
            
        # Accuracy comparison
        accuracy_metrics = {
            'Model': ['Diffusion', 'Discriminative'],
            'Accuracy': [metrics['diffusion_accuracy'], metrics['discriminative_accuracy']]
        }

        df_accuracy = pd.DataFrame(accuracy_metrics)

        plt.figure(figsize=(8, 5))
        sns.barplot(x='Model', y='Accuracy', data=df_accuracy)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        wandb.log({"accuracy_comparison": wandb.Image(plt)})
        plt.close()
    except Exception as e:
        print(f"Warning: Failed to log comparison charts to WandB: {e}")
        # Save locally as fallback
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Model', y='Accuracy', data=df_accuracy)
        plt.title('Accuracy Comparison')
        plt.ylim(0, 1)
        plt.savefig("accuracy_comparison.png")
        plt.close()

def create_thesis_tables(metrics):
    """
    Create formatted tables for thesis use.
    
    Args:
        metrics: Dict containing evaluation metrics
        
    Returns: 
        Dict containing dataframes that can be exported for thesis.
    """
    try:
        # Initialize wandb if not already initialized
        if wandb.run is None:
            wandb.init(project="OOD_diffusion_detector", name="model_comparison")
        
        # 1. Overall metrics comparison table
        overall_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'AUROC'],
            'Diffusion Model': [
                metrics['diffusion_accuracy'],
                metrics['auroc']
            ],
            'Discriminative Model': [
                metrics['discriminative_accuracy'],
                metrics['auroc']
            ]
        })

        # 2. Confusion matrices in dataframe format
        # Convert confusion matrices directly to DataFrames without specifying labels
        diff_conf_df = pd.DataFrame(metrics['diffusion_conf_matrix'])
        disc_conf_df = pd.DataFrame(metrics['discriminative_conf_matrix'])

        # Return all tables
        tables = {
            'overall_metrics': overall_metrics,
            'diffusion_confusion': diff_conf_df,
            'discriminative_confusion': disc_conf_df,
        }

        # Export tables to CSV for thesis use
        for name, df in tables.items():
            df.to_csv(f"{name}.csv")
            try:
                wandb.log({f"{name}_table": wandb.Table(dataframe=df)})
            except Exception as e:
                print(f"Warning: Failed to log {name} table to WandB: {e}")

        return tables
    except Exception as e:
        print(f"Warning: Failed to create thesis tables: {e}")
        # Still create and save the CSV files
        overall_metrics = pd.DataFrame({
            'Metric': ['Accuracy', 'AUROC'],
            'Diffusion Model': [
                metrics['diffusion_accuracy'],
                metrics['auroc']
            ],
            'Discriminative Model': [
                metrics['discriminative_accuracy'],
                metrics['auroc']
            ]
        })
        diff_conf_df = pd.DataFrame(metrics['diffusion_conf_matrix'])
        disc_conf_df = pd.DataFrame(metrics['discriminative_conf_matrix'])
        
        tables = {
            'overall_metrics': overall_metrics,
            'diffusion_confusion': diff_conf_df,
            'discriminative_confusion': disc_conf_df,
        }
        
        for name, df in tables.items():
            df.to_csv(f"{name}.csv")
            
        return tables 