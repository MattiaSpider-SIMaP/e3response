"""
Tensor Parity Plots
^^^^^^^^^^^^^^^^^^^

Compute NMR tensor scalars and plot parity plots at the end of training.
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import jax.numpy as jnp
import jax
from jax import device_put
import matplotlib.pyplot as plt

from reax import hooks
from reax.lightning import rank_zero

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)
__all__ = ("TensorParityPlots",)


def prediction_vs_ground_truth(y_train, y_train_hat, y_val, y_val_hat, y_test, y_test_hat, title):
    """Plot prediction vs ground truth for train, val and test."""
    fontsize = 12
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(y_train, y_train_hat, s=25, c='#b2df8a', label='Train')
    ax.scatter(y_val, y_val_hat, s=25, c='#1f78b4', label='Validation')
    ax.scatter(y_test, y_test_hat, s=25, c="#ff7f0e", label='Test')

    all_obs = jnp.concatenate([y_train, y_val, y_test])
    ax.plot([all_obs.min(), all_obs.max()],
            [all_obs.min(), all_obs.max()],
            'k:', lw=1.5)

    ax.set_aspect('equal')
    ax.set_xlabel('Observation', fontsize=fontsize)
    ax.set_ylabel('Prediction', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend(fontsize=fontsize, handletextpad=0.1, borderpad=0.1)

    fig.tight_layout()
    return fig


def calculate_tensor_scalars(tensors):
    scalars = []

    for tensor in tensors:
        frobenius_norm = jnp.linalg.norm(tensor, 'fro')
        symmetric_part = (tensor + tensor.T) / 2
        asymmetric_part = (tensor - tensor.T) / 2

        tensor_cpu = jax.device_put(tensor, device=jax.devices("cpu")[0])
        eigenvalues, eigenvectors = jnp.linalg.eig(tensor_cpu)
        eigenvalues = -jnp.sort(eigenvalues)

        sigma_xx, sigma_yy, sigma_zz = eigenvalues
        delta_sigma = sigma_zz - (sigma_xx + sigma_yy) / 2
        sigma_iso  = (sigma_xx + sigma_yy + sigma_zz) / 3
        eta = (sigma_xx - sigma_yy) / (sigma_zz - sigma_iso) if abs(sigma_zz - sigma_iso) > 1e-6 else 0

        scalars.append({
            'sigma_xx': sigma_xx,
            'sigma_yy': sigma_yy,
            'sigma_zz': sigma_zz,
            'sigma_iso': sigma_iso,
            'frobenius_norm': frobenius_norm,
            'symmetric_part': symmetric_part,
            'asymmetric_part': asymmetric_part,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'delta_sigma': delta_sigma,
            'eta': eta,
        })

    return scalars

def extract_scalars(scalars_list, key):
    return jnp.array([scalars[key] for scalars in scalars_list])

def compute_tensor_scalars_from_dataloader(dataloader, model, parameters, scalar_keys):
    all_ground_truth = {key: [] for key in scalar_keys}
    all_predicted = {key: [] for key in scalar_keys}

    for batch in dataloader:
        batch = device_put(batch)

        mask = batch[0].nodes['mask']
        y = batch[0].nodes['NMR_tensors'][mask]

        predictions = model.apply(parameters, batch[0])
        mask_hat = predictions[0]['mask']
        y_hat = predictions[0]['predicted_NMR_tensors'][mask_hat]

        scalars_gt = calculate_tensor_scalars(y)
        scalars_pred = calculate_tensor_scalars(y_hat)

        for key in scalar_keys:
            all_ground_truth[key].extend(extract_scalars(scalars_gt, key))
            all_predicted[key].extend(extract_scalars(scalars_pred, key))

    return all_ground_truth, all_predicted


class TensorParityPlots(hooks.TrainerListener):
    """Compute scalar quantities from NMR tensors and log parity plots at the end of training."""

    def __init__(self, scalar_keys: Optional[list[str]] = None, log_rank_zero_only: bool = True):
        self.scalar_keys = scalar_keys or [
            "sigma_xx", "sigma_yy", "sigma_zz", "sigma_iso",
            "delta_sigma", "eta", "frobenius_norm",
            "symmetric_part", "asymmetric_part", "eigenvalues"
        ]
        self.log_rank_zero_only = log_rank_zero_only


    def on_fit_end(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """Executed after training ends."""
        rank = trainer.global_rank if trainer.process_count > 1 else None
        if rank is not None and rank > 0:
            return

        _LOGGER.info(rank_zero.rank_prefixed_message(
            "Computing train/val tensor parity plots...", rank))

        mod = trainer._module

        self.train_gt, self.train_pred = compute_tensor_scalars_from_dataloader(
            trainer.train_dataloader, mod._model, mod.parameters(), self.scalar_keys
        )

        self.val_gt, self.val_pred = compute_tensor_scalars_from_dataloader(
            trainer.val_dataloaders, mod._model, mod.parameters(), self.scalar_keys
        )


    def on_test_end(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """Executed after testing ends."""
        rank = trainer.global_rank if trainer.process_count > 1 else None
        if rank is not None and rank > 0:
            return

        _LOGGER.info(rank_zero.rank_prefixed_message(
            "Computing test tensor parity plots...", rank))

        mod = trainer._module
        test_dl = getattr(stage, "dataloader", None)

        if test_dl is None:
            raise RuntimeError("Test dataloader not found in stage.")

        all_y = []
        all_y_hat = []

        # Manual loop on test dataloader
        for batch in test_dl:
            batch = device_put(batch)

            mask = batch[0].nodes['mask']
            y = batch[0].nodes['NMR_tensors'][mask]

            predictions = mod._model.apply(mod.parameters(), batch[0])
            mask_hat = predictions[0]['mask']
            y_hat = predictions[0]['NMR_tensors_predicted'][mask_hat]

            all_y.append(y)
            all_y_hat.append(y_hat)

        y_test = jnp.concatenate(all_y, axis=0)
        y_hat_test = jnp.concatenate(all_y_hat, axis=0)

        rmse_test = jnp.sqrt(jnp.mean((y_hat_test - y_test) ** 2))

        # Log global rmse
        if trainer.loggers:
            logger = trainer.loggers[0]
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_metric"):
                logger.experiment.log_metric(
                    logger.run_id,
                    "test/stupid/nmr_tensors_rmse",
                    float(rmse_test)
                )
                _LOGGER.info(f"Logged test/nmr_tensors_rmse: {float(rmse_test):.4f}")

        # Compute scalars
        test_gt, test_pred = compute_tensor_scalars_from_dataloader(
            test_dl, mod._model, mod.parameters(), self.scalar_keys
        )

        rmse_per_key = {}

        for key in self.scalar_keys:
            gt = jnp.array(test_gt[key])
            pred = jnp.array(test_pred[key])

            rmse = jnp.sqrt(jnp.mean((gt - pred) ** 2)).real
            rmse_per_key[key] = float(rmse)

            title = f"Parity plot for {key} (Test RMSE = {rmse:.4f})"

            fig = prediction_vs_ground_truth(
                jnp.array(self.train_gt[key]),
                jnp.array(self.train_pred[key]),
                jnp.array(self.val_gt[key]),
                jnp.array(self.val_pred[key]),
                jnp.array(test_gt[key]),
                jnp.array(test_pred[key]),
                title=title,
            )

            if trainer.loggers:
                logger = trainer.loggers[0]
                if hasattr(logger, "experiment") and hasattr(logger.experiment, "log_figure"):
                    logger.experiment.log_figure(logger.run_id, fig, f"{key}_parity.png")
                    _LOGGER.info(rank_zero.rank_prefixed_message(
                        f"Logged figure for {key}.", rank))

                if hasattr(logger.experiment, "log_metric"):
                    logger.experiment.log_metric(
                        logger.run_id,
                        f"test/scalars_rmse/{key}",
                        rmse
                    )
