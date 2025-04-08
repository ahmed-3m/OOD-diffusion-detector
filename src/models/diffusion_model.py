import torch
import diffusers
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import lightning as L
from sklearn.metrics import confusion_matrix

from ..datasets.cifar_dataset import test_cifar_dataset
from ..evaluation.classifier import classify_image


class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Use standard UNet without conditioning since we're only modeling airplanes
        self.model = diffusers.UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D", 
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        # Use the same scheduler as in training
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        self.test_dataloader = self.get_test_dataloader()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Loads a DiffusionModel from checkpoint.
        Assumes checkpoint is a dict with keys "state_dict".
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = cls()
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    def batch_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)

        residual = self.model(
            noisy_images,
            steps,
        ).sample

        loss = F.mse_loss(residual, noise)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.batch_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample_images(self, batch_size, num_inference_steps=50):
        """
        Generate images using a full iterative denoising loop.
        """
        device = self.device

        # Start from pure noise
        sample = torch.randn(batch_size, 3, 32, 32, device=device)

        # Set the scheduler timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps)

        # Iteratively denoise the sample
        for t in self.scheduler.timesteps:
            # Create a batch of current timestep indices
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict the noise residual for the current sample
            noise_pred = self.model(sample, t_batch).sample

            # Get the previous sample using the scheduler's step function
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
        return sample

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        val_loss = self.batch_step(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            # Get images from batch (all airplanes)
            images = batch["images"]
            
            # Generate the same number of airplane images
            num_samples = min(6, images.size(0))
            generated_images = self.sample_images(batch_size=num_samples, num_inference_steps=50)
            
            # Take only the images we need
            true_airplanes = images[:num_samples]
            generated_airplanes = generated_images[:num_samples]
            
            # Combine with true images above, generated below
            combined_display = torch.cat([true_airplanes, generated_airplanes], dim=0)
            combined_grid = torchvision.utils.make_grid(
                combined_display.clamp(-1, 1) * 0.5 + 0.5,
                nrow=num_samples
            )

            # Log combined samples to WandB
            self.logger.experiment.log({
                "val/airplane_samples": wandb.Image(combined_grid, caption="Top: True Airplanes | Bottom: Generated Airplanes")
            })
        return val_loss

    def get_test_dataloader(self):
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
            return images, labels

        data_len = 1000  # Increased for better statistical significance
        prob_dist = [0.5, 0.5]  # Balanced classes
        
        # Set fixed seed for reproducible evaluation
        np.random.seed(42)
        dataset = test_cifar_dataset(data_len, prob_dist)
        np.random.seed(None)  # Reset seed
        
        return torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, 
                                          num_workers=4, collate_fn=collate_fn)

    @torch.no_grad()
    def evaluate_model(self):
        self.eval()  # Ensure model is in evaluation mode
        test_loader = self.test_dataloader
        all_true_labels = []
        all_preds = []
        all_scores = []
        epsilon = 1e-8  # Small constant for numerical stability

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            for i in range(len(images)):
                single_image = images[i]
                true_label = labels[i].item()
                
                # Get prediction and errors for each class
                pred_label, errors = classify_image(self, single_image)
                
                # Calculate confidence scores with numerical stability
                max_error = errors.max()
                if max_error > epsilon:
                    scores = 1.0 - (errors / (max_error + epsilon))
                else:
                    scores = np.ones_like(errors)
                
                all_true_labels.append(true_label)
                all_preds.append(pred_label)
                all_scores.append(scores[1]-scores[0])  # Score for class 1 (not airplane)

        # Convert lists to numpy arrays
        all_true_labels = np.array(all_true_labels)
        all_preds = np.array(all_preds)
        all_scores = np.array(all_scores)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        accuracy = accuracy_score(all_true_labels, all_preds)
        
        # Handle edge case where we only have one class in the labels
        if len(np.unique(all_true_labels)) > 1:
            auroc = roc_auc_score(all_true_labels, all_scores)
        else:
            auroc = 0.5  # Default for single-class case
            
        conf_matrix = confusion_matrix(all_true_labels, all_preds)

        self.train()  # Return to training mode
        return {
            'accuracy': accuracy,
            'auroc': auroc,
            'confusion_matrix': conf_matrix
        }

    def on_train_epoch_end(self):
        # Set model to evaluation mode
        current_epoch = self.current_epoch
        
        # Check if it's a multiple of 10
        if (current_epoch + 1) % 10 == 0:
            self.eval()
            
            with torch.no_grad():
                # 1. Generate and log sample images
                images, labels = next(iter(self.test_dataloader))
                labels = labels.to(self.device)
                generated_images = self.sample_images(batch_size=labels.shape[0], num_inference_steps=50)
                
                # Create visualization grid
                airplane_mask = labels == 0
                airplane_images = generated_images[airplane_mask][:6] if any(airplane_mask) else generated_images[:6]
                other_images = generated_images[~airplane_mask][:6] if any(~airplane_mask) else generated_images[:6]
                combined_images = torch.cat([airplane_images, other_images], dim=0)
                grid = torchvision.utils.make_grid(
                    combined_images.clamp(-1, 1) * 0.5 + 0.5,
                    nrow=6
                )
                self.logger.experiment.log({
                    "test_generated_samples": wandb.Image(grid, caption="Top: Airplanes | Bottom: Other")
                })
                
                # 2. Evaluate classification performance
                metrics = self.evaluate_model()
                
                # 3. Log metrics
                self.logger.experiment.log({
                    "test/accuracy": metrics['accuracy'],
                    "test/auroc": metrics['auroc']
                })
                
                # 4. Create and log confusion matrix visualization
                plt.figure(figsize=(8, 6))
                conf_matrix = metrics['confusion_matrix']
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Airplane", "Not Airplane"], 
                        yticklabels=["Airplane", "Not Airplane"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Classification Confusion Matrix')
                self.logger.experiment.log({"test/confusion_matrix": wandb.Image(plt)})
                plt.close()
            
            # Return to training mode
            self.train()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler] 