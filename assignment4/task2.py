# task2: Autoencoder 2D embedding

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#shallow autoencoder with 2 hidden units
def build_autoencoder(input_dim=784, lr=1e-3):
    #initialize keras layers and models
    layers = tf.keras.layers
    initializers = tf.keras.initializers
    Model = tf.keras.Model
    optimizers = tf.keras.optimizers

    #input layer
    input_layer = layers.Input(shape=(input_dim,))

    #ENCORDER(hidden layer)
    encoded = layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer=initializers.RandomUniform(minval=-0.7, maxval=0.7)
    )(input_layer)

    #DECODER (output layer)
    decoded = layers.Dense(input_dim, activation="linear")(encoded)

    #models complete: input -> encoded -> decoded
    autoencoder = Model(inputs=input_layer, outputs=decoded) #training: reconstruct image
    encoder = Model(inputs=input_layer, outputs=encoded) #find point to plot

    # Compile
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse"
    )

    return autoencoder, encoder


#train autoencoder: X_train -> X_train(regression, MSE)
def train_autoencoder(autoencoder, X_train, epochs=80, batch_size=256, val_split=0.1):
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        verbose=1
    )
    return history


#plot 2D embeddings
def plot_embeddings(embeddings, y_test, classes):
    plt.figure(figsize=(7, 6))
    for c in classes:
        mask = (y_test == c)
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            label=f"Digit {c}",
            alpha=0.7,
            s=10
        )
    plt.title("encoder output: 2D Embedding of MNIST Digits")
    plt.xlabel("Hidden unit 1")
    plt.ylabel("Hidden unit 2")
    plt.legend()
    plt.grid(True)
    plt.show()

#pipeline for Task 2: build autoencoder, training, embeddings extraction, plot
def run_task2(X_train, X_test, y_test, classes, lr=1e-3, epochs=80, batch_size=256, target_mse=0.07, max_restarts=5):
    
    input_dim = X_train.shape[1]
    best = None  # (final_train_mse, autoencoder, encoder, history)

    for attempt in range(1, max_restarts + 1):
        print(f"\nAttempt {attempt}/{max_restarts}: building autoencoder")
        autoencoder, encoder = build_autoencoder(input_dim=input_dim, lr=lr)

        print("Training autoencoder")
        history = train_autoencoder(
            autoencoder,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            val_split=0.1
        )

        #evaluate final MSE
        train_mse = float(history.history["loss"][-1])
        val_mse = float(history.history["val_loss"][-1]) if "val_loss" in history.history else None

        #results
        msg = f"Final train MSE: {train_mse:.4f}"
        if val_mse is not None:
            msg += f" Final val MSE: {val_mse:.4f}"
        print(msg)

        if best is None or train_mse < best[0]:
            best = (train_mse, autoencoder, encoder, history)

        if train_mse <= target_mse:
            print(f"Target reached ({target_mse}).")
            break

    # Use best run
    _, best_autoencoder, best_encoder, _ = best

    if best[0] > target_mse: # did not reach target
        print(f"Warning: best MSE {best[0]:.4f} did not reach target {target_mse}")

    embeddings = best_encoder.predict(X_test, verbose=0)

    plot_embeddings(embeddings, y_test, classes)
