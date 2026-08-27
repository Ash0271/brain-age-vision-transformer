# Brain Age Prediction using a Hybrid CNN + Vision Transformer

Research project from a summer research internship at the USC Viterbi School of
Engineering (IUSSTF&ndash;Viterbi Fellowship, May 2025&ndash;Jan 2026), investigating
hybrid CNN&ndash;Vision Transformer architectures for predicting brain age from
structural MRI (UK Biobank).

## Approach

A 3D CNN stem extracts local features from the MRI volume, which are then patch-embedded,
given a learned positional encoding and classification token, and passed through a stack
of Transformer encoder blocks (multi-head self-attention + MLP) to regress a single
age value. Trained end-to-end with configurable loss functions (MSE, Smooth L1, Huber,
and a custom Reverse Huber loss), optimizers, and LR schedulers, evaluated primarily on
MAE (mean absolute error, in years).

## Files

- `vision_transformer_walkthrough.ipynb` &mdash; an educational, step-by-step build of the
  model (patch embedding, positional encoding, attention, transformer encoder, full model)
  with an example training run and loss curves.
- `model.py` &mdash; the final, configurable version of the model and training script
  (CLI args for architecture/optimizer/scheduler/loss choices), used for the systematic
  hyperparameter sweep.
- `run_experiments.py` &mdash; orchestrates a full hyperparameter sweep from a config CSV,
  automatically resolving "use the best value found so far" for any parameter, logging
  each run, and collecting results.

## Data

The UK Biobank structural MRI data and derived age labels used for training are **not
included** in this repository (subject to UK Biobank's data use agreement). The code
expects a local folder of preprocessed `.nii.gz` volumes and a CSV of participant ages
to run; neither is provided here.

## Status

This project was conducted under a USC lab and was discontinued (advisor relationship
ended) before submission to a venue; ablation studies and further tuning shown in
`run_experiments.py`'s design were in progress at that point. Only my own model
implementation and training code are included here.
