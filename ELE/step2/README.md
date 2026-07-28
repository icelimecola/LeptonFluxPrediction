# Electron Step 2: Weighted LSTM Validation

This folder contains the second-stage LSTM training and drawing workflow.
Its local outputs are written under `step2/Data/` and `step2/Figure/`.

## Baseline Data

From `LeptonFluxPrediction/Electron/step2/`, the old linear-interpolation
baseline can be regenerated with:

```bash
python dataproc_fluxele.py
```

This writes baseline flux products to `step2/Data/flux/`.

## Use Step 1 Imputed Flux

Generate weighted-LSTM hyperparameter YAMLs with paths to the first-stage
products:

```bash
python lstm_hyperpara.py --flux-source imputed
```

This writes these keys into each `step2/Data/hyperpara/paras_*.yaml`:

```yaml
flux_path: ../step1/Data/flux/electron_flux_sun_imputed.npy
error_path: ../step1/Data/flux/electron_err_sun_imputed.npy
```

Then train weighted LSTM as usual from `step2/`:

```bash
bash runtrainw.sh
```

After all jobs finish, select the best weighted model:

```bash
python lstm_bestmodel_w.py
```

## Draw With Step 1 Data

`lstm_draw_w.py` does not read the hyperparameter YAML. To draw the second-stage
model against the first-stage imputed flux/error, run:

```bash
ELECTRON_FLUX_PATH=../step1/Data/flux/electron_flux_sun_imputed.npy \
ELECTRON_ERROR_PATH=../step1/Data/flux/electron_err_sun_imputed.npy \
python lstm_draw_w.py
```

If these environment variables are omitted, drawing uses the baseline files in
`step2/Data/flux/`.

For cluster drawing through the wrapper, use:

```bash
FLUX_SOURCE=imputed bash rundraww.sh
```
