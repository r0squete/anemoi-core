# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

from anemoi.training.diagnostics.callbacks.plot import BasePlotAdditionalMetrics
from anemoi.training.diagnostics.plots import plot_predicted_multilevel_flat_sample

if TYPE_CHECKING:
    from typing import Any

    from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


class DownscalingPlotSample(BasePlotAdditionalMetrics):
    """Plots a downscaling sample: upsampled input (ERA5), target (WRF truth), and prediction.
    
    This callback is specifically designed for downscaling tasks where:
    - Target is at timestep 0 (same-time downscaling, not forecasting)
    - Predictions are already denormalized by add_interp_to_state
    - Batch is raw (not normalized)
    - Input is upsampled via the model's residual layer (grid-agnostic)
    """

    def __init__(
        self,
        config: OmegaConf,
        sample_idx: int,
        parameters: list[str],
        accumulation_levels_plot: list[float],
        precip_and_related_fields: list[str] | None = None,
        colormaps: dict | None = None,
        per_sample: int = 6,
        every_n_batches: int | None = None,
        dataset_names: list[str] | None = None,
        focus_area: list[dict] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the DownscalingPlotSample callback.

        Parameters
        ----------
        config : OmegaConf
            Config object
        sample_idx : int
            Sample to plot
        parameters : list[str]
            Parameters to plot
        accumulation_levels_plot : list[float]
            Accumulation levels to plot
        precip_and_related_fields : list[str] | None, optional
            Precip variable names, by default None
        colormaps : dict | None, optional
            Dictionary of colormaps, by default None
        per_sample : int, optional
            Number of plots per sample (3 for input/truth/pred), by default 6
        every_n_batches : int, optional
            Batch frequency to plot at, by default None
        dataset_names : list[str] | None, optional
            Target dataset names (e.g., ["out_hres"]), by default None
        focus_area : list[dict] | None, optional
            Focus area specification, by default None
        """
        del kwargs
        super().__init__(config, dataset_names=dataset_names, every_n_batches=every_n_batches, focus_area=focus_area)
        self.sample_idx = sample_idx
        self.parameters = parameters
        self.precip_and_related_fields = precip_and_related_fields
        self.accumulation_levels_plot = accumulation_levels_plot
        self.per_sample = per_sample
        self.colormaps = colormaps

        LOGGER.info(
            "Using defined accumulation colormap for fields: %s",
            self.precip_and_related_fields,
        )

    def process(
        self,
        pl_module: pl.LightningModule,
        dataset_name: str,
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process downscaling outputs for plotting.
        
        Returns input (upsampled ERA5), truth (WRF), and prediction (all in physical space).
        
        Parameters
        ----------
        pl_module : pl.LightningModule
            Lightning module
        dataset_name : str
            Target dataset name (e.g., "out_hres")
        outputs : tuple
            (loss, list[dict[dataset: prediction_tensor]])
        batch : dict[str, torch.Tensor]
            Raw batch (not normalized)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (input_upsampled, truth, prediction) all in physical space
        """
        if self.latlons is None:
            self.latlons = {}

        if dataset_name not in self.latlons:
            self.latlons[dataset_name] = pl_module.model.model._graph_data[dataset_name][
                pl_module.model.model._graph_name_data
            ].x.detach()
            self.latlons[dataset_name] = np.rad2deg(self.latlons[dataset_name].cpu().numpy())

        # Truth: raw WRF at timestep 0, select output channels only
        truth_tensor = batch[dataset_name][
            :,
            0 : pl_module.n_step_output,  # timestep 0 for same-time downscaling
            ...,
            pl_module.data_indices[dataset_name].data.output.full,
        ].detach().cpu()[self.sample_idx, 0, ...]  # shape: (grid, vars)

        # Prediction: already denormalized by add_interp_to_state, extract from outputs
        pred_tensor = outputs[1][0][dataset_name].detach().cpu()[
            self.sample_idx, 0, ...
        ]  # shape: (grid, vars) or (ens, grid, vars)
        
        # Handle ensemble dimension if present
        if pred_tensor.ndim == 3:  # (ens, grid, vars)
            pred_tensor = pred_tensor[0, ...]  # Take first member for plotting

        # Input: upsample ERA5 via the model's residual layer (grid-agnostic)
        # This works for both SkipConnection (Option A) and InterpolationConnection (Option B)
        x_in_lres = batch["in_lres"][:, : pl_module.n_step_input].detach()
        input_upsampled = pl_module.model.model.residual["in_lres"](
            x_in_lres,
            grid_shard_shapes=None,
            model_comm_group=None,
        )[
            self.sample_idx, 0, ...
        ].cpu()  # shape: (grid, all_vars), timestep 0, no ensemble dim initially

        # Select channels matching the plotted parameters
        # Get channel indices for the output parameters we're plotting
        output_channel_indices = [
            pl_module.data_indices[dataset_name].model.output.name_to_index[param]
            for param in self.parameters
        ]
        
        # Map from in_lres to out_hres channels (assumes same ordering for shared vars)
        # Use get_matching_channel_indices if available, otherwise assume 1:1 mapping
        if hasattr(pl_module.model.model, 'get_matching_channel_indices'):
            matching_indices = pl_module.model.model.get_matching_channel_indices(dataset_name).cpu()
            input_upsampled = input_upsampled[..., matching_indices]
        else:
            # Fallback: assume in_lres and out_hres share variable ordering
            input_upsampled = input_upsampled[..., output_channel_indices]

        # Apply output mask to truth and prediction
        truth_tensor = pl_module.output_mask[dataset_name].apply(
            truth_tensor,
            dim=pl_module.grid_dim,
            fill_value=np.nan,
        ).numpy()
        
        pred_tensor = pl_module.output_mask[dataset_name].apply(
            pred_tensor,
            dim=pl_module.grid_dim,
            fill_value=np.nan,
        ).numpy()

        input_upsampled = input_upsampled.numpy()

        return input_upsampled, truth_tensor, pred_tensor

    @rank_zero_only
    def _plot(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        dataset_names: list[str],
        outputs: tuple[torch.Tensor, list[dict[str, torch.Tensor]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        epoch: int,
        output_times: tuple,
    ) -> None:
        """Generate downscaling plots.
        
        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer
        pl_module : pl.LightningModule
            Lightning module
        dataset_names : list[str]
            List of target dataset names
        outputs : tuple
            Model outputs
        batch : dict[str, torch.Tensor]
            Input batch
        batch_idx : int
            Batch index
        epoch : int
            Current epoch
        output_times : tuple
            Not used for downscaling (single timestep)
        """
        logger = trainer.logger

        for dataset_name in dataset_names:
            # Build dictionary of indices and parameters to be plotted
            diagnostics = (
                []
                if self.config.data.datasets[dataset_name].diagnostic is None
                else self.config.data.datasets[dataset_name].diagnostic
            )
            plot_parameters_dict = {
                pl_module.data_indices[dataset_name].model.output.name_to_index[name]: (
                    name,
                    name not in diagnostics,
                )
                for name in self.parameters
            }

            input_upsampled, truth, prediction = self.process(pl_module, dataset_name, outputs, batch)

            local_rank = pl_module.local_rank

            # Apply spatial mask
            latlons, input_upsampled, truth, prediction = self.focus_mask.apply(
                pl_module.model.model._graph_data,
                self.latlons[dataset_name],
                input_upsampled,
                truth,
                prediction,
            )

            # Single timestep plot: input (ERA5 upsampled), truth (WRF), prediction
            fig = plot_predicted_multilevel_flat_sample(
                plot_parameters_dict,
                self.per_sample,
                latlons,
                self.accumulation_levels_plot,
                input_upsampled,  # x: upsampled ERA5
                truth,            # y_true: WRF truth
                prediction,       # y_pred: model prediction
                datashader=self.datashader_plotting,
                precip_and_related_fields=self.precip_and_related_fields,
                colormaps=self.colormaps,
            )

            self._output_figure(
                logger,
                fig,
                epoch=epoch,
                tag=f"downscaling_val_sample_{dataset_name}_batch{batch_idx:04d}_rank{local_rank:01d}{self.focus_mask.tag}",
                exp_log_tag=f"val_downscaling_sample_{dataset_name}_rank{local_rank:01d}{self.focus_mask.tag}",
            )
