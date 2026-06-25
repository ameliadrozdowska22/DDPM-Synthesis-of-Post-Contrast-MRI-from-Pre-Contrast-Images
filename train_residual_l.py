import os
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
import dataset
import model
import json

import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from monai.networks.schedulers import DDPMScheduler
from monai.metrics import SSIMMetric

def save_loss_history(train_losses, val_losses, val_loss_epochs, train_loss_epochs, metrics_history=None):
    """
    This function saves loss history
    """
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_loss_epochs": val_loss_epochs,
        "train_loss_epochs": train_loss_epochs,
        "metrics_history": metrics_history or {},
    }
    path = os.path.join(config.RESULTS_DIR, f"models_{config.CURRENT_TRAINING}/loss_history.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f)
    os.replace(tmp, path)


def save_snapshot(inputs, gen_residual, debug_dir, epoch, true_residual, pre_center, post):
    """
    This function saves debug/monitoring visualization, including the
    reconstructed post-contrast image and the residual maps
    """
    generated_post = pre_center + gen_residual * config.RESIDUAL_SCALE

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle(f"Epoch {epoch + 1}", fontsize=14)

    axes[0,0].imshow(inputs[0, 1].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0,0].set_title("condition")
    axes[0,0].axis("off")

    axes[0,1].imshow(generated_post[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0,1].set_title("reconstructed post-contrast")
    axes[0,1].axis("off")

    axes[0,2].imshow(post[0, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0,2].set_title("ground truth")
    axes[0,2].axis("off")

    res_vmax = torch.quantile(true_residual.abs().detach().cpu(), 0.99).item()
    res_vmax = max(res_vmax, 1e-4)

    axes[1,0].imshow(gen_residual[0, 0].cpu(), cmap="bwr", vmin=-res_vmax, vmax=res_vmax)
    axes[1,0].set_title("generated residual")
    axes[1,0].axis("off")

    axes[1,1].axis("off")

    axes[1,2].imshow(true_residual[0, 0].cpu(), cmap="bwr", vmin=-res_vmax, vmax=res_vmax)
    axes[1,2].set_title("target residual")
    axes[1,2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(debug_dir, f"epoch_{epoch+1:04d}.png"))
    plt.close()


def gen_snapshot(targets, scheduler, inputs, unet, debug_dir, epoch, device, pre_center, post):
    """
    This function performs full 1000 step reverse diffusion process on a slice
    to evaluate metrics and visualize for training monitoring
    """
    torch.cuda.empty_cache()
    sample = torch.randn_like(targets)
    scheduler.set_timesteps(num_inference_steps=1000)
    for t in scheduler.timesteps:
        ts = torch.full((inputs.shape[0],), t, device=device, dtype=torch.long)
        with autocast("cuda", enabled=(device.type == "cuda")):
            noise_pred = unet(torch.cat([sample, inputs], dim=1), ts)
        sample = scheduler.step(noise_pred.float(), t, sample)[0]  # reverse diffusion step
    save_snapshot(inputs, sample, debug_dir, epoch, targets, pre_center, post)
    return sample


def save_checkpoint(model, optimizer, epoch, val_noise_loss, path, epochs_without_improvement=0, best_snapshot_mae=None, train_losses=None, val_losses=None, val_loss_epochs=None, train_loss_epochs=None, metrics_history=None, scaler=None, lr_scheduler=None):
    """
    This function saves the training state
    """
    tmp = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_noise_loss": val_noise_loss,
        "best_snapshot_mae": best_snapshot_mae,
        "epochs_without_improvement": epochs_without_improvement,
        "train_losses": train_losses or [],
        "val_losses": val_losses or [],
        "val_loss_epochs": val_loss_epochs or [],
        "train_loss_epochs": train_loss_epochs or [],
        "metrics_history": metrics_history or {},
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(unet, optimizer, path, scaler=None, lr_scheduler=None):
    """
    This function loads training state
    """
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=config.DEVICE)
        unet.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_noise_loss = checkpoint["val_noise_loss"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        val_loss_epochs = checkpoint.get("val_loss_epochs", [])
        train_loss_epochs = checkpoint.get("train_loss_epochs", [])
        metrics_history = checkpoint.get("metrics_history", {})
        best_snapshot_mae = checkpoint.get("best_snapshot_mae", float("inf"))
        print(f"Resumed from epoch {checkpoint['epoch']+1} with val loss {best_noise_loss:.4f}, best MAE {best_snapshot_mae:.4f}")
        return start_epoch, best_noise_loss, best_snapshot_mae, epochs_without_improvement, train_losses, val_losses, val_loss_epochs, train_loss_epochs, metrics_history
    return 0, float("inf"), float("inf"), 0, [], [], [], [], {}



def train_diffusion(device, tl, vl, diffusion_ckpt_path, last_ckpt_path):
    """
    This function serves as a main skeleton for training the diffusion model
    in residual-learning mode
    """
    unet = model.get_model_diff().to(device)

    # MONAI DDPMScheduler performs the forward and reverse DDPM processes
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=True,
        clip_sample_min=-1.0 / config.RESIDUAL_SCALE,
        clip_sample_max=1.0 / config.RESIDUAL_SCALE,
    )

    optimizer = torch.optim.Adam(unet.parameters(), lr=config.LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    start_epoch = 0
    best_noise_loss = float("inf")
    best_snapshot_mae = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_loss_epochs = []
    train_loss_epochs = []
    metrics_history = {}  # {epoch: {"mae": float, "ssim": float, "psnr": float}}

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # if the last training run stopped in the middle of the training we can
    # resume the training from the last checkpoint
    if config.RESUME_TRAINING:
        start_epoch, best_noise_loss, best_snapshot_mae, epochs_without_improvement, train_losses, val_losses, val_loss_epochs, train_loss_epochs, metrics_history = load_checkpoint(unet, optimizer, last_ckpt_path, scaler, lr_scheduler)
        metrics_history = {k: v for k, v in metrics_history.items() if int(k) == 0 or (int(k) + 1) % 10 == 0}
        if os.path.exists(diffusion_ckpt_path):
            best_ckpt = torch.load(diffusion_ckpt_path, map_location=device)
            best_noise_loss = best_ckpt["val_noise_loss"]
            best_snapshot_mae = best_ckpt.get("best_snapshot_mae", float("inf"))

    max_epochs = config.NUM_EPOCHS
    val_interval = config.VAL_INTERVAL

    # directory to store images to monitor the training
    debug_dir = os.path.join(config.RESULTS_DIR, "models_residual_learning/debug_snapshots")
    os.makedirs(debug_dir, exist_ok=True)

    # setting up one fixed validation example for debug/monitoring visualization
    fixed_batch = next(iter(vl))
    fixed_inputs     = fixed_batch[0][:1].to(device)
    fixed_targets    = fixed_batch[1][:1].to(device)
    fixed_pre_center = fixed_batch[2][:1].to(device)
    fixed_post       = fixed_batch[3][:1].to(device)

    for epoch in range(start_epoch, max_epochs):
        unet.train()
        epoch_loss = 0.0

        for step, batch in enumerate(tl):
            inputs, targets, _, _ = batch
            inputs = inputs.to(device)    # [B, 3, H, W] # condition tensor - 3 adjacent pre-contrast slices
            targets = targets.to(device)  # [B, 1, H, W] # target residual (post - pre) / RESIDUAL_SCALE

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                noise = torch.randn_like(targets)  # sample random noise
                timesteps = torch.randint(
                    0,
                    scheduler.num_train_timesteps,
                    (targets.shape[0],),
                    device=device,
                ).long()

                noisy_targets = scheduler.add_noise(
                    original_samples=targets,
                    noise=noise,
                    timesteps=timesteps,
                )  # forward diffusion process

                # if random.random() < config.CONDITION_TURN_OFF:
                #     inputs = torch.zeros_like(inputs)

                model_input = torch.cat([noisy_targets, inputs], dim=1)  # [B, 4, H, W] input concatenated with condition

                noise_pred = unet(model_input, timesteps)  # predict noise
                loss = F.mse_loss(noise_pred.float(), noise.float())


            # optimizer update
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # training monitoring
            epoch_loss += loss.item()
            if step % 10 == 0 or step == len(tl) - 1:
                print(
                    f"Epoch {epoch + 1}/{max_epochs} | "
                    f"Step {step + 1}/{len(tl)} | "
                    f"Loss: {epoch_loss / (step + 1):.4f} | "
                    f"MSE: {loss.item():.4f} | "
                )

        train_losses.append(epoch_loss / (step + 1))
        train_loss_epochs.append(epoch)

        # validation
        if (epoch + 1) % val_interval == 0:
            unet.eval()
            val_noise_loss = 0.0

            # flag indicating whether to visualize and evaluate the debug/monitoring slice
            # done every 10 epochs
            do_snapshot = epoch == 0 or (epoch + 1) % 10 == 0

            with torch.no_grad():

                # perform the full 1000 step reverse diffusion process on the debug slice
                # to visualize and evaluate
                if do_snapshot:
                    sample = gen_snapshot(fixed_targets, scheduler, fixed_inputs, unet, debug_dir, epoch, device, fixed_pre_center, fixed_post)

                    # reconstruct post-contrast image from the generated residual, then
                    # calculate visual evaluation metrics on it
                    generated_post = fixed_pre_center + sample * config.RESIDUAL_SCALE
                    snapshot_mae = torch.mean(torch.abs(generated_post - fixed_post)).item()
                    mse = torch.mean((generated_post - fixed_post) ** 2).item()
                    psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else 100.0
                    ssim = ssim_metric(generated_post, fixed_post).mean().item()
                    ssim_metric.reset()

                    # save the metrics for monitoring
                    metrics_history[str(epoch)] = {"mae": snapshot_mae, "ssim": ssim, "psnr": psnr}
                    print(f"Image metrics: MAE: {snapshot_mae:.4f} | SSIM: {ssim:.4f} | PSNR: {psnr:.2f} dB")

                _val_step = 0
                for _val_step, batch in enumerate(vl, start=1):
                    inputs, targets, _, _ = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    with autocast("cuda", enabled=(device.type == "cuda")):
                        noise = torch.randn_like(targets)  # sample random noise
                        timesteps = torch.randint(
                            0,
                            scheduler.num_train_timesteps,
                            (targets.shape[0],),
                            device=device,
                        ).long()

                        noisy_targets = scheduler.add_noise(
                            original_samples=targets,
                            noise=noise,
                            timesteps=timesteps,
                        )  # forward diffusion process

                        model_input = torch.cat([noisy_targets, inputs], dim=1)  # [B, 4, H, W] input concatenated with condition
                        noise_pred = unet(model_input, timesteps)  # predict noise

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_noise_loss += loss.item()

            if _val_step > 0:
                val_noise_loss /= _val_step  # average val loss for the epoch
            val_losses.append(val_noise_loss)
            val_loss_epochs.append(epoch)
            print(f"Epoch {epoch + 1} val loss: {val_noise_loss:.4f}")
            lr_scheduler.step(val_noise_loss)  # learning rate decrease
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  LR: {current_lr:.2e}")

            # save new best checkpoint if current val loss significantly lower than last best
            if val_noise_loss < best_noise_loss - config.MIN_DELTA:
                best_noise_loss = val_noise_loss
                if do_snapshot:
                    best_snapshot_mae = snapshot_mae
                epochs_without_improvement = 0
                save_checkpoint(unet, optimizer, epoch, val_noise_loss, diffusion_ckpt_path, best_snapshot_mae=best_snapshot_mae, train_losses=train_losses, val_losses=val_losses, val_loss_epochs=val_loss_epochs, train_loss_epochs=train_loss_epochs, metrics_history=metrics_history, lr_scheduler=lr_scheduler)
                print(f"New best model saved (val loss: {best_noise_loss:.4f})" + (f" | MAE: {best_snapshot_mae:.4f}" if do_snapshot else ""))
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

            # save last training state (in case we would like to resume training)
            save_checkpoint(unet, optimizer, epoch, val_noise_loss, last_ckpt_path, epochs_without_improvement, best_snapshot_mae=best_snapshot_mae, train_losses=train_losses, val_losses=val_losses, val_loss_epochs=val_loss_epochs, train_loss_epochs=train_loss_epochs, metrics_history=metrics_history, scaler=scaler, lr_scheduler=lr_scheduler)
            save_loss_history(train_losses, val_losses, val_loss_epochs, train_loss_epochs, metrics_history)

            # stop if no validation loss improvements for enough epochs (early stopping)
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print("done")


if __name__ == "__main__":
    config.CURRENT_TRAINING = "residual_learning"
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # path to best training state
    diffusion_ckpt_path = os.path.join(config.RESULTS_DIR, "models_residual_learning/best_diffusion.pth")

    # path to resumable (last) training state
    last_ckpt_path = os.path.join(config.RESULTS_DIR, "models_residual_learning/diffusion_checkpoint.pth")

    print(f"Device: {config.DEVICE}")
    print("Loading data...")
    tl, vl, _ = dataset.get_dataloaders(include_test=False)

    print(f"I'm training {config.CURRENT_TRAINING}")
    print("Building model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_diffusion(device, tl, vl, diffusion_ckpt_path, last_ckpt_path)
