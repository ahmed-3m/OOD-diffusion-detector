import os
import argparse
import torch
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Import from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.diffusion_model import DiffusionModel
from src.datasets.data_module import DiffusionData

def main(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set up the directory paths
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    
    # Initialize model and data
    model = DiffusionModel()
    data = DiffusionData(batch_size=args.batch_size)
    
    # Set up WandB logger
    wandb_mode = "offline" if args.offline else None
    wandb_logger = WandbLogger(
        save_dir=base_dir,
        project="OOD_diffusion_detector",
        name=args.experiment_name,
        mode=wandb_mode
    )
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=base_dir,
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True
    )
    
    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        default_root_dir=base_dir,
        devices=args.devices,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.gradient_clip_val,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy=args.strategy
    )
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Train the model
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        trainer.fit(model, data, ckpt_path=args.checkpoint_path)
    else:
        print("Starting new training run")
        trainer.fit(model, data)
    
    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")
    
    # Finish WandB logging
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model for OOD detection")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1501, help="Number of training epochs")
    parser.add_argument("--accumulate_grad_batches", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision for training")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--strategy", type=str, default="auto", help="Training strategy")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    
    # Logging parameters
    parser.add_argument("--experiment_name", type=str, default="diffusion_ood_detector", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--log_every_n_steps", type=int, default=25, help="Log every N steps")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--offline", action="store_true", help="Run WandB in offline mode")
    
    # Checkpointing
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    main(args) 