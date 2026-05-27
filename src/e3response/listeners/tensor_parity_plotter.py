import pathlib
from typing import Any, Final

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing_extensions import override

from tensorial.reaxkit.listeners.parity_plotter import GraphParityPlotter
from tensorial.reaxkit.utils import pylogger

__all__ = ("TensorGraphParityPlotter",)

_LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)


class TensorGraphParityPlotter(GraphParityPlotter):
    """
    Computes selected properties from true/predicted tensors and generates a parity plot for each.
    """
    def __init__(
        self,
        targets: str = "nodes.nmr_tensors",
        predictions: str = "nodes.predicted_nmr_tensors",
        properties_keys: list[str] | None = None,
        save_dir: str | pathlib.Path = "parity_plots",
        fit_plot_every: int = 100,
        show_rmse: bool = False,
        **kwargs
    ):
        super().__init__(
            targets=targets, 
            predictions=predictions, 
            save_dir=save_dir, 
            fit_plot_every=fit_plot_every, 
            **kwargs
        )

        self.show_rmse = show_rmse
        self.properties_keys = properties_keys or [
            "sigma xx",
            "sigma yy",
            "sigma zz",
            "sigma iso",
            "delta sigma",
            "eta",
            "frobenius norm",
            "symmetric part",
            "antisymmetric part",
            "eigenvalues",
        ]       

    def _compute_all_properties(self, tensors: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Compute all requested properties."""
        results = {}
        
        if "symmetric part" in self.properties_keys:
            # (T + T.T) / 2
            results["symmetric part"] = (tensors + tensors.swapaxes(-1, -2)) / 2.0
            
        if "antisymmetric part" in self.properties_keys:
            # (T - T.T) / 2
            results["antisymmetric part"] = (tensors - tensors.swapaxes(-1, -2)) / 2.0

        sym_tensors = (tensors + tensors.swapaxes(-1, -2)) / 2.0
        
        # Check if autoval are needed
        needs_eig = any(k in self.properties_keys for k in [
            "sigma xx", "sigma yy", "sigma zz", "sigma iso", 
            "delta sigma", "eta", "eigenvalues"
        ])

        if needs_eig:
            eigvals = jnp.linalg.eigvalsh(sym_tensors)
            #  zz >= yy >= xx
            eigvals_sorted = jnp.sort(eigvals, axis=-1)[..., ::-1] 
            
            s_zz = eigvals_sorted[..., 0]
            s_yy = eigvals_sorted[..., 1]
            s_xx = eigvals_sorted[..., 2]
            s_iso = (s_zz + s_yy + s_xx) / 3.0

            if "eigenvalues" in self.properties_keys:
                results["eigenvalues"] = eigvals_sorted
            if "sigma zz" in self.properties_keys:
                results["sigma zz"] = s_zz
            if "sigma yy" in self.properties_keys:
                results["sigma yy"] = s_yy
            if "sigma xx" in self.properties_keys:
                results["sigma xx"] = s_xx
            if "sigma iso" in self.properties_keys:
                results["sigma iso"] = s_iso
            if "delta sigma" in self.properties_keys:
                # Δσ = σzz - (σxx + σyy)/2
                results["delta sigma"] = s_zz - (s_xx + s_yy) / 2.0
            if "eta" in self.properties_keys:
                # η = (σxx - σyy) / (σzz - σiso)
                denominator = s_zz - s_iso
                results["eta"] = jnp.where(jnp.abs(denominator) > 1e-6, (s_xx - s_yy) / denominator, 0.0)

        if "frobenius norm" in self.properties_keys:
            results["frobenius norm"] = jnp.linalg.norm(tensors, axis=(-2, -1))
            
        return results

    @override
    def _collect_batch_data(self, stage_name: str, outputs: Any | None, batch: Any) -> bool:
        """
        Modified function to collect tensors.
        """
        if outputs is None: return False

        tensors_gt, tensors_pred = super()._get_target_predicted(batch, outputs)
        
        # gt_dict = self._compute_all_properties(jnp.array(tensors_gt))
        # pred_dict = self._compute_all_properties(jnp.array(tensors_pred))

        self.data_store[stage_name][0].append(np.array(tensors_gt))
        self.data_store[stage_name][1].append(np.array(tensors_pred))
        
        return True

    @override
    def _plot_parity(self, stage_name: str, save_dir: pathlib.Path, epoch: int | None = None):
        raw_gt_list, raw_pred_list = self.data_store[stage_name]
        if not raw_gt_list:
            return
        
        all_tensors_gt = np.concatenate(raw_gt_list, axis=0)
        all_tensors_pred = np.concatenate(raw_pred_list, axis=0)

        gt_dict = self._compute_all_properties(jnp.array(all_tensors_gt))
        pred_dict = self._compute_all_properties(jnp.array(all_tensors_pred))

        for key in self.properties_keys:
            y_true = np.array(gt_dict[key]).flatten()
            y_pred = np.array(pred_dict[key]).flatten()

            title = f"{key} ({stage_name.capitalize()} Stage)"
            if self.show_rmse:
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                title += f" | RMSE: {rmse:.4f}"

            fig, ax = plt.subplots(figsize=(8, 8))
            self._scatter_parity(
                ax, y_true, y_pred, label=f"{stage_name} points (N={len(y_true)})"
            )
            self._finalize_parity_ax(
                ax,
                y_true,
                y_pred,
                x_label=f"True {key}",
                y_label=f"Predicted {key}",
                title=title,
            )

            (save_dir / key).mkdir(parents=True, exist_ok=True)
            filename = f"{stage_name}_epoch_{epoch}.pdf" if epoch is not None else f"{stage_name}.pdf"
            plt.savefig(str(save_dir / key / filename), bbox_inches="tight")
            plt.close(fig)

    @override
    def _plot_combined_all_stages(self, trainer):
        """
        Override for tensors: computes properties once per stage 
        and generates the final combined parity plots.
        """
        # 1. Pre-calcoliamo le proprietà per ogni stage presente in data_store
        computed_stages = {}
        for stage, _, _ in self.COLOR_CFG:
            raw_gt_list, raw_pred_list = self.data_store[stage]
            if raw_gt_list: # Se ci sono dati accumulati per questo stage
                # Concatena i batch grezzi NumPy
                all_gt = np.concatenate(raw_gt_list, axis=0)
                all_pred = np.concatenate(raw_pred_list, axis=0)
                
                # Calcola le proprietà una sola volta per lo stage attuale usando JAX
                computed_stages[stage] = {
                    "gt": self._compute_all_properties(jnp.array(all_gt)),
                    "pred": self._compute_all_properties(jnp.array(all_pred))
                }

        # 2. Ora cicliamo sulle proprietà per fare i grafici combinati finali
        for key in self.properties_keys:
            stage_data = {}
            
            for stage in computed_stages.keys():
                # Estraiamo gli array già calcolati dal dizionario e facciamo il flatten
                y_true = np.array(computed_stages[stage]["gt"][key]).flatten()
                y_pred = np.array(computed_stages[stage]["pred"][key]).flatten()
                stage_data[stage] = (y_true, y_pred)
            
            if stage_data:
                save_dir = self._get_save_dir(trainer) / "combined_final"
                
                title = key
                if self.show_rmse:
                    rmse_strs = []
                    for stage, (y_t, y_p) in stage_data.items():
                        rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
                        rmse_strs.append(f"{stage.capitalize()} RMSE: {rmse:.4f}")
                    title += " | " + " - ".join(rmse_strs)
                
                # Rispettiamo l'assegnazione delle etichette dinamiche
                self._x_label = f"True {key}"
                self._y_label = f"Predicted {key}"
                
                self._plot_combined_data(
                    stage_data, 
                    save_dir, 
                    filename=f"{key.replace(' ', '_')}_combined_final.pdf",
                    title=title
                )