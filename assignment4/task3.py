# task3: image inpainting with autoencoder (parametric masking)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#---------------------------------------------------------------------MASKING FUNCTIONS
#functions to apply horizontal masks
def mask_horizontal(img, start_row, height, value=0.0):
    masked = img.copy()
    masked[start_row:start_row + height, :] = value
    return masked

#functions to apply vertical masks
def mask_vertical(img, start_col, width, value=0.0):
    masked = img.copy()
    masked[:, start_col:start_col + width] = value
    return masked

#functions to apply patch masks
def mask_patch(img, top, left, h, w, value=0.0):
    masked = img.copy()
    masked[top:top + h, left:left + w] = value
    return masked

#apply a mask to an image according to the specified mode and severity
def apply_mask(img, mode="mix", severity=0.3, value=0.0, rng=None):
    """
    mode: 'h', 'v', 'p', or 'mix'
    severity: approx fraction of pixels to mask (0..1)
    """

    #select random mode
    if rng is None:
        rng = np.random.default_rng()

    H, W = img.shape  # 28, 28
    mode_eff = mode
    if mode == "mix":
        mode_eff = rng.choice(["h", "v", "p"])

    #convert severity into sizes: area to mask = severity * (H*W)
    target_area = int(np.clip(severity, 0.0, 1.0) * H * W)

    if mode_eff == "h":
        #mask some full-width rows. height = target_area / W
        height = max(1, min(H, target_area // W))
        start = rng.integers(0, H - height + 1)
        return mask_horizontal(img, start, height, value=value)

    elif mode_eff == "v":
        #mask some full-height cols. width = target_area / H
        width = max(1, min(W, target_area // H))
        start = rng.integers(0, W - width + 1)
        return mask_vertical(img, start, width, value=value)

    else:  # 'p'
        #patch roughly square: h*w = target_area
        side = int(np.sqrt(max(1, target_area)))
        h = max(1, min(H, side))
        w = max(1, min(W, max(1, target_area // h)))
        top = rng.integers(0, H - h + 1)
        left = rng.integers(0, W - w + 1)
        return mask_patch(img, top, left, h, w, value=value)


#create a masked dataset with the same size as X: X_flat(N, 784) -> X_masked_flat N, 784)
def create_masked_dataset(X_flat, mode="mix", severity=0.3, value=0.0, seed=42):
    rng = np.random.default_rng(seed)
    X_masked = np.empty_like(X_flat, dtype=np.float32)

    for i, x in enumerate(X_flat):
        img = x.reshape(28, 28)
        masked = apply_mask(img, mode=mode, severity=severity, value=value, rng=rng)
        X_masked[i] = masked.reshape(784)

    return X_masked


#---------------------------------------------------------------------AUTOENCODER MODEL
#function to build the inpainting autoencoder model
def build_inpainting_autoencoder(input_dim=784, lr=1e-3):
    #initialize keras layers, model, and optimizers
    layers = tf.keras.layers
    Model = tf.keras.Model
    optimizers = tf.keras.optimizers

    #build model
    inp = layers.Input(shape=(input_dim,))

    #encoder
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    #decoder
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(input_dim, activation="sigmoid")(x)

    autoencoder = Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse")
    return autoencoder


#results: original, masked, reconstructed image
def show_reconstruction(original, masked, reconstructed, n=6, title=None):
    plt.figure(figsize=(10, 6))
    if title:
        plt.suptitle(title)

    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Original")

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(masked[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Masked")

        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
        if i == 0:
            ax.set_title("Reconstructed")

    plt.tight_layout()
    plt.show()


#pipeline to run Task 3: train and evaluate inpainting autoencoder
def run_task3(X_train, X_test, mode="mix", severities=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), epochs=20, batch_size=256, mask_value=0.0, seed=42):

    #initialize input dimension
    input_dim = X_train.shape[1]
    results = []  #to store results

    for sev in severities:
        print(f"\nmode={mode} severity={sev:.2f}")

        X_train_masked = create_masked_dataset(X_train, mode=mode, severity=sev, value=mask_value, seed=seed)
        X_test_masked = create_masked_dataset(X_test, mode=mode, severity=sev, value=mask_value, seed=seed + 1)

        model = build_inpainting_autoencoder(input_dim=input_dim, lr=1e-3)

        print("Training")
        model.fit(
            X_train_masked, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.1,
            verbose=1
        )

        print("Evaluating")
        test_mse = model.evaluate(X_test_masked, X_test, verbose=0)
        print(f"Test MSE: {test_mse:.4f}")
        results.append({               
            "mode": mode,
            "severity": sev,
            "test_mse": test_mse
        })

        recon = model.predict(X_test_masked, verbose=0)
        show_reconstruction(
            X_test, X_test_masked, recon,
            n=6,
            title=f"Inpainting | mode={mode} | severity={sev:.2f} | test_mse={test_mse:.4f}"
        )

        print("\nResults:")
        print(f"{'Severity':<10} {'Test MSE':<10} {'Quality':<10}");
        print("-" * 32)

        for r in results:
            if r['test_mse'] > 0.09: quality = "Poor" 
            elif r['test_mse'] > 0.045: quality = "Okay" 
            elif r['test_mse'] > 0.02: quality = "Good" 
            else: quality = "Excellent"

            print(f"{r['severity']:<10.2f} {r['test_mse']:<10.4f} {quality:<10}")