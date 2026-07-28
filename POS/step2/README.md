# Positron Step 2: Weighted LSTM Validation

This folder contains the second-stage LSTM training and drawing workflow.
Its local outputs are written under `step2/Data/` and `step2/Figure/`.

## Baseline Data

From `LeptonFluxPrediction/Positron/step2/`, the old linear-interpolation
baseline can be regenerated with:

```bash
python dataproc_fluxpos.py
```

This writes baseline flux products to `step2/Data/flux/`.

The positron ROOT file has 29 raw energy bins. This workflow drops bin 0
`[0.8, 1.0] GeV` and the last bin `[41.9, 45.1] GeV`, keeping 27 bins from
`1.0` to `41.9 GeV`.

## Use Step 1 Imputed Flux

This mode requires positron Step 1 products to exist under `../step1/Data/flux/`.
Until those products are generated, use the baseline files from `dataproc_fluxpos.py`.

Generate weighted-LSTM hyperparameter YAMLs with paths to the first-stage
products:

```bash
python lstm_hyperpara.py --flux-source imputed
```

This writes these keys into each `step2/Data/hyperpara/paras_*.yaml`:

```yaml
flux_path: ../step1/Data/flux/positron_flux_sun_imputed.npy
error_path: ../step1/Data/flux/positron_err_sun_imputed.npy
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
POSITRON_FLUX_PATH=../step1/Data/flux/positron_flux_sun_imputed.npy \
POSITRON_ERROR_PATH=../step1/Data/flux/positron_err_sun_imputed.npy \
python lstm_draw_w.py
```

If these environment variables are omitted, drawing uses the baseline files in
`step2/Data/flux/`.

For cluster drawing through the wrapper, use:

```bash
FLUX_SOURCE=imputed bash rundraww.sh
```
