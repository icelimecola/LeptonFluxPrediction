# Positron Step 2.2: Sun-Only Weighted LSTM Baseline

This folder is a control workflow for the positron project. It keeps the
step2 weighted-LSTM training/drawing structure, but the model input is only
the five solar parameters. The labels are the linearly interpolated positron
flux products generated locally in `step2.2/Data/flux/`.

## Run Order

Run from `LeptonFluxPrediction/Positron/step2.2/`:

```bash
python dataproc_fluxpos.py
python lstm_hyperpara.py
bash runtrainw.sh
python lstm_bestmodel_w.py
python lstm_draw_w.py
```

For cluster drawing through the wrapper:

```bash
bash rundraww.sh
```

Optional solar-lag diagnostic figures:

```bash
python draw_fluxsunpara.py
```

This creates faceted time-series plots, z-score overlays, lag-correlation
curves, best-lag flux/solar scatter plots, and a lag summary table in
`Figure/fluxsunpara/`.

## Outputs

- `Data/flux/`: linearly interpolated positron flux/error and energy edges
- `Data/modelw/`: `sunOnlyWeighted_*.keras`, `best_model.txt`, `model_summary.csv`
- `Data/lstmdraww/`: train/validation/test relative-error arrays
- `Figure/lstmtrainw/`: loss curves
- `Figure/lstmdraww/`: prediction/error PDF and HTML plots
- `Figure/fluxsunpara/`: solar-parameter lag diagnostic plots

## Notes

- This workflow does not read step1 sun-imputed flux.
- The positron ROOT file has 29 raw energy bins. This workflow drops bin 0
  `[0.8, 1.0] GeV` and the last bin `[41.9, 45.1] GeV`, keeping 27 bins from
  `1.0` to `41.9 GeV`.
- The weighted loss is retained from step2; only the input features change from
  `solar + flux + err` to `solar`.
