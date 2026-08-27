#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import pandas as pd
import os
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.nn import MSELoss, L1Loss, SmoothL1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Check for HuberLoss (PyTorch 1.9+)
if hasattr(nn, 'HuberLoss'):
    from torch.nn import HuberLoss
else:
    class HuberLoss(nn.Module):
        def __init__(self, delta=1.0, reduction='mean'):
            super().__init__()
            self.delta = delta
            self.reduction = reduction
        def forward(self, input, target):
            return F.huber_loss(input, target, delta=self.delta, reduction=self.reduction)

# ===========================
# Custom Loss Functions
# ===========================
class ReverseHuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        loss = torch.where(
            abs_error <= self.delta,
            abs_error,
            (abs_error**2 + self.delta**2) / (2 * self.delta)
        )
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# ===========================
# Argparse setup
# ===========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    # NOTE: cnn_layers is kept for compatibility with run_experiments.py filenames, 
    # but the architecture is now FIXED (4 stages).
    parser.add_argument("--cnn_layers", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--patch_size", type=str, default="(8,8,8)")
    parser.add_argument("--n_channels", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.0025)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--eta_min", type=float, default=1e-7)
    parser.add_argument("--loss_fn", type=str, default="MSELoss")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    return parser.parse_args()

# ===========================
# CNN Block (NEW)
# ===========================
class ConvStage(nn.Module):
    def __init__(self, in_ch, out_ch, downsample):
        super().__init__()
        # If downsample=True, stride=2 (halves the size). Else stride=1.
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            # Second conv always has stride 1 to refine features at this scale
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# ===========================
# Patch Embedding
# ===========================
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()
        self.linear_project = nn.Conv3d(n_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.linear_project(x)
        x = x.flatten(2).transpose(-2, -1)
        return x

# ===========================
# Positional Encoding
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(d_model):
                pe[pos][i] = np.sin(pos / (10000 ** (i / d_model))) if i % 2 == 0 else np.cos(pos / (10000 ** ((i - 1) / d_model)))
        self.register_buffer("pe_base", pe.unsqueeze(0))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        B, P, D = x.shape
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        seq_len = x.size(1)
        if seq_len > self.pe_base.size(1):
            pe = F.interpolate(self.pe_base.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False).permute(0, 2, 1)
        else:
            pe = self.pe_base[:, :seq_len, :]
        return x + pe

# ===========================
# Transformer Core
# ===========================
class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.scale = head_size ** 0.5

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        return attn @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList([AttentionHead(d_model, head_size) for _ in range(n_heads)])
        self.W_o = nn.Linear(n_heads * head_size, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.W_o(out)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4, dropout=0.3):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*r_mlp, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.dropout(self.mha(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x

# ===========================
# Vision Transformer (UPDATED)
# ===========================
class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, cnn_layers=8, dropout=0.3):
        super().__init__()
        
        # --- NEW FIXED ARCHITECTURE ---
        # Note: 'cnn_layers' arg is ignored here to enforce the new robust structure.
        self.cnn_stem = nn.Sequential(
            ConvStage(n_channels, 8, downsample=False), # Size: 128 -> 128
            ConvStage(8, 16, downsample=True),          # Size: 128 -> 64
            ConvStage(16, 32, downsample=True),         # Size: 64 -> 32
            ConvStage(32, 64, downsample=True),         # Size: 32 -> 16
        )
        
        # We downsampled 3 times (factor of 2 each time).
        # Total downsample factor = 2 * 2 * 2 = 8.
        downsample_factor = 8 
        cnn_output_shape = tuple(s // downsample_factor for s in img_size)
        
        # The final ConvStage outputs 64 channels
        cnn_channels_out = 64
        
        # --- End of CNN updates ---

        self.patch_embedding = PatchEmbedding(d_model, cnn_output_shape, patch_size, cnn_channels_out)
        n_patches = (cnn_output_shape[0]*cnn_output_shape[1]*cnn_output_shape[2]) // (patch_size[0]*patch_size[1]*patch_size[2])
        self.positional_encoding = PositionalEncoding(d_model, n_patches+1)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(d_model, n_heads) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.cnn_stem(x)
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x[:, 0])
        return self.classifier(x)

# ===========================
# Dataset
# ===========================
class PreloadedBrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_csv):
        self.samples = []
        df = pd.read_csv(label_csv)
        label_map = dict(zip(df['Participant ID'].astype(str),
                             df['Age when attended assessment centre | Instance 2']))
        for fname in os.listdir(root_dir):
            if fname.endswith(".nii.gz"):
                sid = fname.split("_")[0]
                if sid not in label_map:
                    continue
                age = label_map[sid]
                img = nib.load(os.path.join(root_dir, fname)).get_fdata()
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-5)
                self.samples.append((img, torch.tensor(age, dtype=torch.float32), sid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ===========================
# Training Script
# ===========================
def main():
    args = parse_args()
    d_model = args.d_model
    patch_size = eval(args.patch_size)
    
    # We keep cnn_layers for file naming, but the model class ignores it.
    cnn_layers = args.cnn_layers 
    
    batch_size = args.batch_size
    epochs = args.epochs
    alpha = args.alpha
    weight_decay = args.weight_decay
    eta_min = args.eta_min
    n_heads = args.n_heads
    n_layers = args.n_layers
    n_channels = args.n_channels

    label_csv_path = "Age_Acq_merged.csv"
    full_dataset = PreloadedBrainDataset(root_dir="./downsampled", label_csv=label_csv_path)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pass cnn_layers just to satisfy arguments, though it is unused inside.
    transformer = VisionTransformer(d_model, 1, (128,128,128), patch_size, n_channels, n_heads, n_layers, cnn_layers).to(device)

    # Dynamic optimizer
    optimizer_class = AdamW if args.optimizer == "AdamW" else Adam
    optimizer = optimizer_class(transformer.parameters(), lr=alpha, weight_decay=weight_decay)

    # Loss Selection Logic
    loss_fn_norm = args.loss_fn.lower().replace(" ", "").replace("_", "")
    
    if "mse" in loss_fn_norm:
        criterion = MSELoss()
    elif "smoothl1" in loss_fn_norm:
        criterion = SmoothL1Loss()
    elif "reversehuber" in loss_fn_norm:
        criterion = ReverseHuberLoss()
    elif "huber" in loss_fn_norm:
        criterion = HuberLoss()
    else:
        # Default fallback
        criterion = L1Loss()
        
    print(f"Selected Loss Function: {criterion}")

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min) if args.scheduler == "CosineAnnealingLR" \
                 else ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # === Dynamic file naming ===
    base_name = f"run_{args.run_id}_cnn{cnn_layers}_dm{d_model}_bs{batch_size}_e{epochs}"
    checkpoint_dir = f"checkpoints_{base_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = open(f"train_log_{base_name}.txt", "w")

    train_losses, val_losses, train_maes, val_maes, lrs = [], [], [], [], []
    best_val_mae, best_epoch = float("inf"), 0

    for epoch in range(epochs):
        transformer.train()
        train_loss, train_abs_err, total_samples = 0, 0, 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = transformer(inputs).squeeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_abs_err += torch.sum(torch.abs(outputs - labels)).item()
            total_samples += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_abs_err / total_samples

        transformer.eval()
        val_loss, val_abs_err, val_samples = 0, 0, 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = transformer(inputs).squeeze(1)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                val_abs_err += torch.sum(torch.abs(outputs - labels)).item()
                val_samples += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_abs_err / val_samples

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch + 1

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        torch.save(transformer.state_dict(), f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")

        log_line = (f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.2f} | "
                    f"Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.2f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.12f}")
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)
        lrs.append(optimizer.param_groups[0]['lr'])

    log_file.close()

    csv_name = f"epoch_metrics_{base_name}.csv"
    pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_mae": train_maes,
        "val_mae": val_maes,
        "current_lr": lrs
    }).to_csv(csv_name, index=False)

    print(f"\nBest Epoch: {best_epoch}")
    print(f"Best Val MAE: {best_val_mae:.4f}")

if __name__ == "__main__":
    main()