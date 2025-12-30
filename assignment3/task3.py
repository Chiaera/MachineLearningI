import numpy as np
from pathlib import Path
import keras as tf_keras
import matplotlib.pyplot as plt
from mainTask import (
    load_data,
    minmax_normalize_train, minmax_X,
    minmax_normalize_train_Y, minmax_Y
)



def build_model(input_dim, output_dim, h):
    model = tf_keras.Sequential(name='shallow_network')
    model.add(tf_keras.layers.Dense(h, activation='sigmoid', input_shape=(input_dim,)))
    model.add(tf_keras.layers.Dense(output_dim, activation='linear'))
    
    model.compile(optimizer='adam', loss=tf_keras.losses.MeanSquaredError())
    
    return model

#plot
def plot_history(history, title):
    plt.figure()
    plt.plot(history['loss'], label='train loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='val loss')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(True)
    plt.show()


#--------------------------------------------------------------------------------------
def main():
   #dataset
    directory = Path(__file__).resolve().parent
    data_path = directory/"data"/"data.txt"
    X, Y = load_data(path=data_path, c=2, shuffle=True)
    n, d = X.shape
    c = Y.shape[1]

    #folds
    kfolds = 5
    num_trials = 5 
    num_epochs = 30
    batch_size = 100

    h_values = [4, 8, 12, 16, 20, 40]
    
    lperc, hperc = 25, 75

    results = []
    all_histories = []

    for h in h_values:
        print(f"\nTesting h = {h}")

        fold_mse_best = [] 

        #cross-validation
        for k in range(kfolds):

            print(f"\nfold {k+1}/{kfolds} ---")

            idx_min = int(n * k / kfolds)
            idx_max = int(n * (k + 1) / kfolds)

            # test
            X_test = X[idx_min:idx_max]
            Y_test = Y[idx_min:idx_max]

            # train
            idx_train = list(range(idx_min)) + list(range(idx_max, n))
            X_train = X[idx_train]
            Y_train = Y[idx_train]

            #normalization
            X_train_n, minX, rangeX = minmax_normalize_train(X_train)
            X_test_n = minmax_X(X_test, minX, rangeX)

            Y_train_n, minY, rangeY = minmax_normalize_train_Y(Y_train)
            Y_test_n = minmax_Y(Y_test, minY, rangeY)

            #multi-start
            best_mse_fold = None
            fold_histories = [] 

            for trial in range(num_trials):

                print(f"Trial {trial+1}/{num_trials}")

                model = build_model(d, c, h)

                #train
                history = model.fit(
                    X_train_n, Y_train_n,
                    validation_split=0.1,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1
                )

                #test MSE 
                Y_pred = model.predict(X_test_n, verbose=0)
                mse = ((Y_pred - Y_test_n)**2).mean()
                print(f"mse = {mse:.3E}")

                history_dict = {
                    "h":h,
                    "fold": k,
                    "trial": trial,
                    "loss": history.history["loss"],
                    "val_loss": history.history.get("val_loss", None),
                    "mse": mse
                }

                if best_mse_fold is None or mse < best_mse_fold:
                    best_mse_fold = mse

                fold_histories.append(history_dict)

            all_histories.extend(fold_histories)
            fold_mse_best.append(best_mse_fold)


        #resume
        p25, median, p75 = np.percentile(fold_mse_best, (lperc, 50, hperc))
        ratio = (p75 - p25) / median if median > 0 else np.inf

        results.append({
            "h": h,
            "median": median,
            "p25": p25,
            "p75": p75,
            "ratio": ratio
        })

    #RESUME
    print("\nRESUME")
    header = "{:>6} {:>14} {:>14} {:>14} {:>14}".format("h", "median", "p25", "p75", "ratio")
    print(header)
    print("-" * len(header))

    for r in results:
        print("{:6d} {:14.3E} {:14.3E} {:14.3E} {:14.3E}".format(r["h"], r["median"], r["p25"], r["p75"], r["ratio"]))


    #best h
    best = min(results, key=lambda rr: (rr["median"], rr["ratio"]))
    h_best = best["h"]
    print(f"\nBest h = {h_best} (median={best['median']:.3E}, ratio={best['ratio']:.3E})")

    hist_best_h = [hh for hh in all_histories if hh["h"] == h_best]
    mse_list_best = [h["mse"] for h in hist_best_h]
    best_idx = np.argmin(mse_list_best)
    worst_idx = np.argmax(mse_list_best)
    best_hist_h_best = hist_best_h[best_idx]
    worst_hist_h_best = hist_best_h[worst_idx]

    #worst h 
    worst = max(results, key=lambda rr: (rr["median"], rr["ratio"]))
    h_worst = worst["h"]
    print(f"Worst h = {h_worst} (median={worst['median']:.3E}, ratio={worst['ratio']:.3E})")

    hist_worst_h = [hh for hh in all_histories if hh["h"] == h_worst]
    mse_list_worst = [h["mse"] for h in hist_worst_h]
    best_idx_worst  = np.argmin(mse_list_worst)
    worst_idx_worst = np.argmax(mse_list_worst)
    best_hist_h_worst  = hist_worst_h[best_idx_worst]
    worst_hist_h_worst = hist_worst_h[worst_idx_worst]

    #plot
    plot_history(best_hist_h_best, f"Best h ({h_best}) - best trial ({best_hist_h_best['trial']})")
    plot_history(worst_hist_h_best, f"Best h ({h_best}) - worst trial ({worst_hist_h_best['trial']})")

    plot_history(best_hist_h_worst, f"Worst h ({h_worst}) - best trial ({best_hist_h_worst['trial']})")
    plot_history(worst_hist_h_worst, f"Worst h ({h_worst}) - worst trial ({worst_hist_h_worst['trial']})")



if __name__ == "__main__":
    main()