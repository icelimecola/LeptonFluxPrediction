# Electron Step 1: Sun-only Flux Imputation

This folder contains the first-stage electron flux imputation workflow.
It writes its own products under `step1/Data/` and `step1/Figure/`.

## Outputs

The first-stage flux/data products are written to `Data/flux/`:

- `electron_flux_observed_nan.npy`: observed electron flux, with missing entries as `NaN`
- `electron_err_observed_nan.npy`: observed electron error, with missing entries as `NaN`
- `electron_observed_mask.npy`: observed mask, `1` for real AMS observations and `0` for missing entries
- `electron_flux_sun_pred_allbin.npy`: sun-only LSTM predictions for every observed-period day
- `electron_flux_sun_imputed.npy`: observed flux preserved, missing flux filled with sun-only predictions
- `electron_err_sun_imputed.npy`: observed error preserved, missing error estimated conservatively

The final draw step also writes overview plots to `Figure/flux/`:

- `electron_flux_sun_imputed_overview.pdf/html`: final continuous flux with imputed points highlighted
- `electron_flux_sun_imputed_overview_points.pdf/html`: observed and imputed points only
- `electron_flux_sun_imputed_overview_with_error.pdf/html`: observed and imputed points with final error bars

It also produces lstmdraw-style first-stage evaluation plots:

- `electron_sun_imputer_prediction_<model>.pdf/html`: observed flux vs sun-only predictions in training/validation/test
- `electron_sun_imputer_error_<model>.pdf/html`: relative error in training/validation/test
- `Data/lstmdraw/train_error_allbin_<model>.npy`, `Data/lstmdraw/val_error_allbin_<model>.npy`, `Data/lstmdraw/test_error_allbin_<model>.npy`

## Run Order

Run these commands from `LeptonFluxPrediction/Electron/step1/`:

```bash
python dataproc_realflux.py
python lstm_train_sunpara.py --config Data/hyperpara/paras_0.yaml
python lstm_draw_sunpara.py
python validate_masked_gaps.py
```

`lstm_draw_sunpara.py` produces both final flux and final err, and writes the
final overview/evaluation plots unless plot options disable them.

Use the Python environment that already runs the existing TensorFlow/uproot scripts.

For hyperparameter scans, use one of these wrappers instead of calling a single
`paras_0.yaml` manually:

```bash
# Local/interactive serial scan over Data/hyperpara/paras_*.yaml
bash runtrain_sun_imputer.sh

# Cluster submission, matching the existing jsub style
bash runtrain_sun_imputer_jsub.sh
```

The local wrapper runs `lstm_bestmodel_sunpara.py` automatically after all
configs finish. For `jsub`, run it manually after all submitted jobs are done:

```bash
python lstm_bestmodel_sunpara.py
```

This writes `Data/model/model_summary.csv` and `Data/model/best_model.txt`.

## Notes

- The sun-only imputer predicts flux only. It does not predict measurement errors.
- Missing labels do not contribute to first-stage loss. They are masked out.
- Missing errors are estimated as the larger of local/interpolated AMS error and a model-error term from validation relative RMSE.
- This is a feasibility validation pipeline. The second stage intentionally treats the imputed flux/error as ordinary complete inputs.
