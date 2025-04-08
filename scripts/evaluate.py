import os
import argparse
import torch
import wandb

# Import from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.diffusion_model import DiffusionModel
from src.evaluation.classifier import (
    get_pretrained_cifar10_model,
    evaluate_models,
    log_confusion_matrices,
    log_comparison_charts,
    create_thesis_tables
)

def main(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb for the evaluation
    try:
        if args.offline:
            print("Running in offline mode - WandB logging will be saved locally")
            wandb.init(project="OOD_diffusion_detector", name=args.experiment_name, mode="offline")
        else:
            wandb.init(project="OOD_diffusion_detector", name=args.experiment_name)
    except Exception as e:
        print(f"Warning: Failed to initialize WandB: {e}")
        print("Continuing without WandB logging")
    
    # Load model checkpoint
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        # Try to find the best checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        best_checkpoint_path = None
        best_val_loss = float('inf')
        
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith("best-") and filename.endswith(".ckpt"):
                    try:
                        # Extract val_loss from filename (best-{epoch:02d}-{val_loss:.2f}.ckpt)
                        val_loss_str = filename.split("-")[-1].split(".ckpt")[0]
                        val_loss = float(val_loss_str)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_checkpoint_path = os.path.join(checkpoint_dir, filename)
                    except (IndexError, ValueError):
                        continue
                        
        # If no best checkpoint found, fall back to last checkpoint
        if best_checkpoint_path is None:
            last_checkpoint_path = os.path.join(checkpoint_dir, "last.ckpt")
            if os.path.exists(last_checkpoint_path):
                checkpoint_path = last_checkpoint_path
                print(f"No best checkpoint found, falling back to last checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"No checkpoint found at {checkpoint_path} or in {checkpoint_dir}")
        else:
            checkpoint_path = best_checkpoint_path
            print(f"Found best checkpoint: {checkpoint_path} (val_loss: {best_val_loss:.4f})")
    
    # Load models
    print(f"Loading diffusion model from checkpoint: {checkpoint_path}")
    diffusion_model = DiffusionModel.load_from_checkpoint(checkpoint_path)
    diffusion_model = diffusion_model.to(device)
    diffusion_model.eval()
    
    print("Loading pretrained discriminative model")
    discriminative_model = get_pretrained_cifar10_model(device)
    
    # Evaluate both models
    print("Starting model evaluation...")
    evaluation_metrics = evaluate_models(
        diffusion_model,
        discriminative_model,
        diffusion_model.test_dataloader,
        num_samples=args.num_samples
    )
    print("Evaluation complete!")

    # Print debugging information
    print("\n===== DEBUG INFORMATION =====")
    for key, value in evaluation_metrics.items():
        if isinstance(value, (torch.Tensor, list, tuple, set, dict)):
            print(f"{key}: {type(value)}, length={len(value)}")
        else:
            print(f"{key}: {value}")
    print("=============================\n")

    # Log results to wandb
    print("Logging results to WandB...")
    log_confusion_matrices(evaluation_metrics)
    log_comparison_charts(evaluation_metrics)

    # Create thesis tables
    print("Creating thesis tables...")
    thesis_tables = create_thesis_tables(evaluation_metrics)

    # Print summary of results
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Diffusion Classifier Accuracy: {evaluation_metrics['diffusion_accuracy']:.4f}")
    print(f"Discriminative Classifier Accuracy: {evaluation_metrics['discriminative_accuracy']:.4f}")
    print(f"AUROC: {evaluation_metrics['auroc']:.4f}")
    print("==============================\n")

    print("Tables have been exported as CSV files for thesis use.")
    
    # Safely finish wandb
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"Warning: Failed to gracefully finish WandB run: {e}")
        print("This is not critical - your results have been saved locally.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diffusion model OOD detector")
    
    # Evaluation parameters
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to checkpoint file or directory containing checkpoints")
    parser.add_argument("--num_samples", type=int, default=1000, 
                        help="Number of samples to evaluate")
    
    # Logging parameters
    parser.add_argument("--experiment_name", type=str, default="model_evaluation", 
                        help="Name of the experiment")
    parser.add_argument("--offline", action="store_true", 
                        help="Run WandB in offline mode")
    
    args = parser.parse_args()
    main(args) 