import numpy as np
import matplotlib.pyplot as plt
import keras as tf_keras
from pathlib import Path
from mainTask import (
    load_data,
    minmax_normalize_train, minmax_X,
    minmax_normalize_train_Y, minmax_Y
)

def build_model(input_dim, output_dim, h):
    model = tf_keras.Sequential(name='shallow_network')
    model.add(tf_keras.layers.Dense(h, activation='sigmoid', input_shape=(input_dim,), name='hidden'))
    model.add(tf_keras.layers.Dense(output_dim, activation='linear', name='output'))
    
    model.compile(optimizer='adam', loss=tf_keras.losses.MeanSquaredError())
    
    return model


#--------------------------------------------------------------------------------------
def main():
    #dataset
    directory = Path(__file__).resolve().parent
    data_path = directory/"data"/"data.txt"
    X, Y = load_data(path=data_path, c=2, shuffle=True)
    n, d = X.shape
    c = Y.shape[1]

    #split
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[n_train:n_train + n_val]
    Y_val = Y[n_train:n_train + n_val]

    X_test = X[n_train + n_val:]
    Y_test = Y[n_train + n_val:]

    #train normalization
    X_train_n, minX, rangeX = minmax_normalize_train(X_train)
    X_val_n = minmax_X(X_val, minX, rangeX)
    X_test_n = minmax_X(X_test, minX, rangeX)

    Y_train_n, minY, rangeY = minmax_normalize_train_Y(Y_train)
    Y_val_n = minmax_Y(Y_val, minY, rangeY)
    Y_test_n = minmax_Y(Y_test, minY, rangeY)

    #shallow neural network
    h = 12              
    num_epochs = 30
    batch_size = 100
    num_trials = 10    

    input_dim = d
    output_dim = c

    best_test_mse = None
    best_model = None

    histories = []      
    test_mse_all = []   

    #multi-start
    for trial in range(num_trials):
        print(f"TRIAL {trial + 1}/{num_trials}")
        model = build_model(input_dim, output_dim, h)

        history = model.fit(
            X_train_n, Y_train_n,
            validation_data=(X_val_n, Y_val_n),
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=1
        )
        histories.append((trial, history.history))

        # test
        Y_pred_test = model.predict(X_test_n, verbose=0)
        mse_test = ((Y_pred_test - Y_test_n) ** 2).mean()
        test_mse_all.append(mse_test)

        print(f"Test MSE (trial {trial + 1}): {mse_test:.4e}")

        if best_test_mse is None or mse_test < best_test_mse:
            best_test_mse = mse_test
            best_model = model

    #results
    print(f"Test: best MSE={best_test_mse:.4e}")
    print(f"Trial: mean MSE={np.mean(test_mse_all):.4e}")
    print(f"Trial: std MSE={np.std(test_mse_all):.4e}")

    #resume"
    print("RESUME")
    print(test_mse_all)
    print("")

    lperc, hperc = 25, 75
    p75, median, p25 = np.percentile(test_mse_all, (hperc, 50, lperc))

    print(f"mse = {median:.3E} (typical), "f"mse in [{p25:.3E}, {p75:.3E}] "f"with probability >= {(hperc - lperc)/100:.2f}")
    
    #plot
    if len(test_mse_all) > 1:
        best_idx = int(np.argmin(test_mse_all))
        worst_idx = int(np.argmax(test_mse_all))

        for label, idx in [("best", best_idx), ("worst", worst_idx)]:
            trial_id, hist = histories[idx]

            plt.figure()
            plt.plot(hist['loss'], label='train loss')
            if 'val_loss' in hist:
                plt.plot(hist['val_loss'], label='val loss')
            plt.title(f"Trial {trial_id} ({label})")
            plt.xlabel("Epoch")
            plt.ylabel("MSE loss")
            plt.legend()
            plt.grid(True)

        plt.show()
    else:
        print("Trials are not enough")


if __name__ == "__main__":
    main()
