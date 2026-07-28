#!/bin/python

import argparse
import random

import numpy as np
import yaml

from common import (
    PAD_DAYS,
    SUN_OFFSET,
    ELECTRON_DIR,
    ensure_dirs,
    find_best_model,
    imputation_path,
    minmax_inverse,
    minmax_transform,
    model_dir,
    nanminmax_fit,
    relative_rmse_by_bin,
    train_figure_dir,
    trainerr_path,
    load_sun_daily,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train sun-only LSTM imputer with masked flux loss.")
    parser.add_argument("--config", type=str, default=str(ELECTRON_DIR / "step1" / "Data" / "hyperpara" / "paras_0.yaml"))
    parser.add_argument("--flux", type=str, default=str(imputation_path("electron_flux_observed_nan.npy")))
    parser.add_argument("--mask", type=str, default=str(imputation_path("electron_observed_mask.npy")))
    return parser.parse_args()


def masked_huber_factory(bins):
    import tensorflow as tf

    def masked_huber(y_packed, y_pred, delta=1.0):
        y_true = y_packed[:, :bins]
        mask = y_packed[:, bins:]
        abs_error = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + delta * linear
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(loss * mask) / denom

    return masked_huber


def make_sequences(X, y, mask, look_back):
    X_seq, y_seq, m_seq = [], [], []
    for i in range(look_back, len(X)):
        X_seq.append(X[i - look_back:i, :])
        y_seq.append(y[i, :])
        m_seq.append(mask[i, :])
    return np.array(X_seq), np.array(y_seq), np.array(m_seq)


def main():
    ensure_dirs()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    look_back = int(config["look_back"])
    train_num = float(config["train_num"])
    val_num = float(config["val_num"])
    neurons = int(config["neurons"])
    l2 = float(config["l2"])
    dropout = float(config["dropout"])
    learning_rate = float(config["learning_rate"])
    batch_size = int(config["batch_size"])
    epoch_begin = int(config["epoch_begin"])
    epochs = int(config["epochs"])
    epoch_end = epoch_begin + epochs

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Dense, Input, LSTM
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adamax
    from tensorflow.keras.regularizers import L1L2

    tf.random.set_seed(seed)

    sun_daily = load_sun_daily()
    flux_obs = np.load(args.flux)
    mask_obs = np.load(args.mask).astype(bool)
    bins = flux_obs.shape[1]

    flux_padded = np.concatenate([np.full((PAD_DAYS, bins), np.nan), flux_obs], axis=0)
    mask_padded = np.concatenate([np.zeros((PAD_DAYS, bins), dtype=bool), mask_obs], axis=0)
    X_all = sun_daily[SUN_OFFSET:SUN_OFFSET + len(flux_padded), 0:5]

    number = len(flux_padded)
    train_end = int(number * train_num)
    val_end = int(number * (train_num + val_num))
    split_tag = f"{train_num}train_{val_num}val_"

    X_train = X_all[:train_end]
    X_val = X_all[train_end - look_back:val_end]
    X_test = X_all[val_end - look_back:number]

    y_train_raw = flux_padded[:train_end]
    y_val_raw = flux_padded[train_end - look_back:val_end]
    y_test_raw = flux_padded[val_end - look_back:number]

    m_train = mask_padded[:train_end]
    m_val = mask_padded[train_end - look_back:val_end]
    m_test = mask_padded[val_end - look_back:number]

    x_min, x_max = nanminmax_fit(X_train)
    y_min, y_max = nanminmax_fit(y_train_raw, m_train)
    X_train = minmax_transform(X_train, x_min, x_max)
    X_val = minmax_transform(X_val, x_min, x_max)
    X_test = minmax_transform(X_test, x_min, x_max)
    y_train = minmax_transform(y_train_raw, y_min, y_max)
    y_val = minmax_transform(y_val_raw, y_min, y_max)
    y_test = minmax_transform(y_test_raw, y_min, y_max)

    X_train_seq, y_train_seq, m_train_seq = make_sequences(X_train, y_train, m_train.astype(float), look_back)
    X_val_seq, y_val_seq, m_val_seq = make_sequences(X_val, y_val, m_val.astype(float), look_back)
    X_test_seq, y_test_seq, m_test_seq = make_sequences(X_test, y_test, m_test.astype(float), look_back)

    y_train_packed = np.concatenate([y_train_seq, m_train_seq], axis=1)
    y_val_packed = np.concatenate([y_val_seq, m_val_seq], axis=1)

    model = Sequential([
        Input(shape=(look_back, 5), dtype="float32"),
        LSTM(units=neurons, dropout=dropout, kernel_regularizer=L1L2(l1=0, l2=l2), name="LSTM"),
        Dense(bins, name="output_flux"),
    ])

    optimizer = Adamax(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=masked_huber_factory(bins), metrics=[], weighted_metrics=[])
    model.summary()

    prefix = (
        "sunImputer_"
        f"{epoch_begin}-{epoch_end}epoch_"
        f"{learning_rate}learningRate_"
        f"{neurons}neurons_"
        f"{l2}l2_"
        f"{dropout}dropout_"
        f"{batch_size}batchSize_"
        f"{split_tag}"
    )
    checkpoint = ModelCheckpoint(
        str(model_dir() / (prefix + "{epoch:04d}-{val_loss:.5f}.keras")),
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        mode="auto",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=100, min_delta=0.00001, mode="min", verbose=1)

    history = model.fit(
        X_train_seq,
        y_train_packed,
        validation_data=(X_val_seq, y_val_packed),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    best_path = find_best_model("sunImputer_", prefer_best_file=False)
    (model_dir() / "best_model.txt").write_text(best_path.name + "\n", encoding="utf-8")

    scaler_payload = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "look_back": look_back,
        "train_num": train_num,
        "val_num": val_num,
    }
    np.savez(trainerr_path("sun_imputer_scalers.npz"), **scaler_payload)

    best_model = load_model(best_path, custom_objects={"masked_huber": masked_huber_factory(bins)})
    val_pred_scaled = best_model.predict(X_val_seq, verbose=0)
    train_pred_scaled = best_model.predict(X_train_seq, verbose=0)
    test_pred_scaled = best_model.predict(X_test_seq, verbose=0)

    val_pred = minmax_inverse(val_pred_scaled, y_min, y_max)
    train_pred = minmax_inverse(train_pred_scaled, y_min, y_max)
    test_pred = minmax_inverse(test_pred_scaled, y_min, y_max)
    val_true = minmax_inverse(y_val_seq, y_min, y_max)
    train_true = minmax_inverse(y_train_seq, y_min, y_max)
    test_true = minmax_inverse(y_test_seq, y_min, y_max)

    val_rrmse = relative_rmse_by_bin(val_pred, val_true, m_val_seq)
    train_rrmse = relative_rmse_by_bin(train_pred, train_true, m_train_seq)
    test_rrmse = relative_rmse_by_bin(test_pred, test_true, m_test_seq)
    np.save(trainerr_path("sun_imputer_validation_relative_rmse_per_bin.npy"), val_rrmse)
    np.save(trainerr_path("sun_imputer_train_relative_rmse_per_bin.npy"), train_rrmse)
    np.save(trainerr_path("sun_imputer_test_relative_rmse_per_bin.npy"), test_rrmse)

    import matplotlib.pyplot as plt

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.yscale("log")
    plt.title("sun-only imputer loss")
    plt.ylabel("masked loss")
    plt.xlabel("epoch")
    plt.legend(["training", "validation"], loc="upper left")
    plt.savefig(train_figure_dir() / ("loss_" + prefix.rstrip("_") + ".pdf"), bbox_inches="tight")
    plt.close()

    print("Best sun imputer:", best_path)
    print("Saved scalers:", trainerr_path("sun_imputer_scalers.npz"))
    print("Validation relative RMSE per bin:", trainerr_path("sun_imputer_validation_relative_rmse_per_bin.npy"))


if __name__ == "__main__":
    main()
