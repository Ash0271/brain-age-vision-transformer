#!/usr/bin/env python3
import pandas as pd
import subprocess
import os
from datetime import datetime

CONFIG_CSV = "documentation.csv"
MODEL_SCRIPT = "VisionTransformerNewMod.py"

# Read CSV, ensuring consistent handling of best_val_mae
df = pd.read_csv(CONFIG_CSV)

# Track best values per column dynamically
best_values = {}

# Columns that can have "BEST" entries
PARAM_COLS = [
    "cnn_layers", "d_model", "patch_size",
    "n_channels", "n_heads", "n_layers",
    "batch_size", "epochs", "alpha", "weight_decay",
    "eta_min", "loss_fn", "optimizer", "scheduler"
]

def resolve_value(param_name, value):
    """Replace 'BEST' with the stored best value."""
    if isinstance(value, str) and value.strip().upper() == "BEST":
        if param_name in best_values:
            print(f"    ⭐ Resolved 'BEST' {param_name} -> {best_values[param_name]}")
            return best_values[param_name]
        else:
            raise ValueError(f"⚠️ 'BEST' used for '{param_name}' before any best value was established.")
    return value

def update_best_values(df):
    """Update the best_values dictionary based on lowest val_mae so far."""
    # Convert best_val_mae to numeric, forcing errors to NaN
    df["best_val_mae"] = pd.to_numeric(df["best_val_mae"], errors='coerce')
    
    # Filter only completed runs
    completed = df.dropna(subset=["best_val_mae"])
    
    if completed.empty:
        return

    # Find row with minimum MAE
    best_idx = completed["best_val_mae"].idxmin()
    best_run = completed.loc[best_idx]
    
    print(f"🏆 Current Best Run: {best_run['run_id']} (MAE: {best_run['best_val_mae']:.4f})")
    
    for col in PARAM_COLS:
        if col in df.columns:
            best_values[col] = best_run[col]

# Initialize with any existing completed runs
update_best_values(df)

# Iterate over each experiment row
for idx, row in df.iterrows():
    run_id = row["run_id"]

    # Check if this run is already completed (has a numeric MAE)
    if not pd.isna(pd.to_numeric(row["best_val_mae"], errors='coerce')):
        print(f"✅ Run {run_id} already completed, skipping.")
        continue

    print(f"\n🚀 Starting Run {run_id}...")

    # Resolve parameters (replace BEST with real values)
    resolved = {}
    try:
        for col in PARAM_COLS:
            if col in row:
                resolved[col] = resolve_value(col, row[col])
    except ValueError as e:
        print(str(e))
        continue

    # Handle tuple conversion for patch_size
    patch_size = str(resolved["patch_size"])
    if isinstance(patch_size, str):
        patch_size = patch_size.replace("(", "").replace(")", "").replace(" ", "")
        # Ensure it's formatted as (x,y,z) string for arg parser or just passed cleanly
        patch_size = f"({patch_size})"

    # Create unique output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"run_{run_id}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Build command
    # NOTE: We still pass cnn_layers here so the model script doesn't crash 
    # and so the file naming remains consistent with your CSV.
    cmd = [
        "python3", MODEL_SCRIPT,
        "--run_id", str(run_id),
        "--cnn_layers", str(resolved["cnn_layers"]),
        "--d_model", str(resolved["d_model"]),
        "--patch_size", patch_size,
        "--n_channels", str(resolved["n_channels"]),
        "--n_heads", str(resolved["n_heads"]),
        "--n_layers", str(resolved["n_layers"]),
        "--batch_size", str(resolved["batch_size"]),
        "--epochs", str(resolved["epochs"]),
        "--alpha", str(resolved["alpha"]),
        "--weight_decay", str(resolved["weight_decay"]),
        "--eta_min", str(resolved["eta_min"]),
        "--optimizer", str(resolved["optimizer"]),
        "--scheduler", str(resolved["scheduler"]),
        "--loss_fn", str(resolved["loss_fn"]),
    ]

    log_path = os.path.join(out_dir, f"train_log_run_{run_id}.txt")

    print(f"    Running command: {' '.join(cmd)}")

    # Run experiment and log output
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Read metrics CSV created by the model
    # Note: Ensure the filename matches exactly what VisionTransformerNewMod.py generates
    metrics_csv_name = f"epoch_metrics_run_{run_id}_cnn{resolved['cnn_layers']}_dm{resolved['d_model']}_bs{resolved['batch_size']}_e{resolved['epochs']}.csv"
    
    if os.path.exists(metrics_csv_name):
        metrics = pd.read_csv(metrics_csv_name)
        best_epoch = metrics["val_mae"].idxmin() + 1
        best_val_mae = metrics["val_mae"].min()

        # Update dataframe in memory
        df.loc[idx, "best_epoch"] = best_epoch
        df.loc[idx, "best_val_mae"] = best_val_mae

        # Update best values dynamically for the NEXT iteration
        update_best_values(df)

        print(f"✅ Run {run_id} completed — Best Epoch {best_epoch}, Val MAE = {best_val_mae:.4f}")
        
        # Move the metrics file to the output folder to keep root clean
        os.rename(metrics_csv_name, os.path.join(out_dir, metrics_csv_name))
        
    else:
        print(f"⚠️ Run {run_id} did not produce a metrics file ({metrics_csv_name}). Check logs.")

    # Save progress after each run to CSV
    df.to_csv(CONFIG_CSV, index=False)

print("\n🎯 All runs completed. Results updated in documentation.csv.")