"""Optimization loops for centered-product CSCAE training."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .pipeline import CenteredProductCSCAENN, metrics, threshold_from_validation


def clip_with_scale(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values / scale, -1, 1).astype(np.float32)


def probabilities(model, values: np.ndarray, device, batch: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(values), batch):
            logits, _ = model(torch.from_numpy(values[start:start + batch]).to(device))
            result.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(result)


def fit_model(
    output: Path,
    name: str,
    ae_train: np.ndarray,
    ae_validation: np.ndarray,
    supervised_train: np.ndarray,
    y_train: np.ndarray,
    supervised_validation: np.ndarray,
    y_validation: np.ndarray,
    r2_test: np.ndarray,
    y_r2_test: np.ndarray,
    config: dict,
) -> dict:
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    product_shape = tuple(map(int, ae_train.shape[1:]))
    if len(product_shape) != 2 or any(tuple(values.shape[1:]) != product_shape for values in (
        ae_validation, supervised_train, supervised_validation, r2_test
    )):
        raise ValueError("all model inputs must share one [j,time] product shape")
    model = CenteredProductCSCAENN(product_shape, config["model"]["latent_ratio"],
                                   config["model"]["channels"], config["model"]["classifier_widths"]).to(device)
    reconstruction = nn.MSELoss()
    batch = int(config["training"]["batch_size"])
    started = time.time()

    train_tensor = torch.from_numpy(np.ascontiguousarray(ae_train, np.float32))
    validation_tensor = torch.from_numpy(np.ascontiguousarray(ae_validation, np.float32))
    optimizer = torch.optim.AdamW(model.autoencoder_parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch, shuffle=True)
    best_state = copy.deepcopy(model.state_dict()); best_loss = float("inf"); stale = 0; ae_history = []
    for epoch in range(1, int(config["training"]["autoencoder_epochs"]) + 1):
        model.train(); total = 0.0
        for (clean,) in loader:
            clean = clean.to(device)
            noisy = torch.clamp(clean + torch.randn_like(clean) * config["training"]["noise_std"], -1, 1)
            optimizer.zero_grad(set_to_none=True)
            rebuilt = model.reconstruct(noisy)
            loss = reconstruction(rebuilt, clean)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"]); optimizer.step()
            total += float(loss.detach()) * len(clean)
        model.eval(); validation_loss = 0.0
        with torch.no_grad():
            for start in range(0, len(validation_tensor), batch):
                clean = validation_tensor[start:start + batch].to(device)
                validation_loss += float(reconstruction(model.reconstruct(clean), clean)) * len(clean)
        row = {"epoch": epoch, "train_reconstruction_loss": total / len(train_tensor),
               "validation_reconstruction_loss": validation_loss / len(validation_tensor)}
        ae_history.append(row)
        if row["validation_reconstruction_loss"] < best_loss:
            best_loss = row["validation_reconstruction_loss"]; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= int(config["training"]["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)

    x_train = torch.from_numpy(np.ascontiguousarray(supervised_train, np.float32))
    labels = torch.from_numpy(y_train.astype(np.int64))
    counts = np.bincount(y_train, minlength=2)
    weights = counts.sum() / np.maximum(2 * counts, 1)
    classification = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    loader = DataLoader(TensorDataset(x_train, labels), batch_size=batch, shuffle=True)
    best_state = copy.deepcopy(model.state_dict()); best_auc = -1.0; stale = 0; classifier_history = []
    for epoch in range(1, int(config["training"]["classifier_epochs"]) + 1):
        model.train(); total = 0.0
        for clean, label in loader:
            clean = clean.to(device); label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, rebuilt = model(clean)
            loss = classification(logits, label) + config["training"]["auxiliary_reconstruction_weight"] * reconstruction(rebuilt, clean)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"]); optimizer.step()
            total += float(loss.detach()) * len(clean)
        validation_probability = probabilities(model, supervised_validation, device, batch)
        score = float(roc_auc_score(y_validation, validation_probability))
        classifier_history.append({"epoch": epoch, "train_joint_loss": total / len(x_train), "validation_auc": score})
        if score > best_auc:
            best_auc = score; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= int(config["training"]["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    validation_probability = probabilities(model, supervised_validation, device, batch)
    threshold = threshold_from_validation(y_validation, validation_probability)
    r2_probability = probabilities(model, r2_test, device, batch)
    result = {
        "name": name,
        "device": str(device),
        "fit_seconds": time.time() - started,
        "autoencoder_history": ae_history,
        "classifier_history": classifier_history,
        "best_validation_auc": best_auc,
        "threshold": threshold,
        "threshold_source": "R0/R7 validation only",
        "r0_r7_validation": metrics(y_validation, validation_probability, threshold),
        "r2_heldout_test": metrics(y_r2_test, r2_probability, threshold),
    }
    torch.save({"schema": "maskedbike-qshare-cscae.v2", "product_shape": product_shape, "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": config, "threshold": threshold, "positive_event": "hw_zero"}, output / "model.pt")
    np.savez_compressed(output / "predictions.npz", r2_probability=r2_probability, r2_truth=y_r2_test,
                        r0r7_validation_probability=validation_probability, r0r7_validation_truth=y_validation)
    return result
