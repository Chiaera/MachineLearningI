import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
from Knn import Knn 

# graphic library
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# --------------------------------------- DEF ---------------------------------------  
def mnist_subsets(X_train, Y_train, X_test, Y_test, train_size=10000, test_size=2000):
    # train
    train_indices = np.random.choice(X_train.shape[0], train_size, replace=False)
    x_train = X_train[train_indices].reshape((train_size, -1)) / 255
    y_train = Y_train[train_indices]

    # test
    test_indices = np.random.choice(X_test.shape[0], test_size, replace=False)
    x_test = X_test[test_indices].reshape((test_size, -1)) / 255
    y_test = Y_test[test_indices]

    return x_train, y_train, x_test, y_test


def pca_plot (dataset, x_train, x_test, y_test, y_pred, title):
    pca = PCA(n_components=2) 
    pca.fit(x_train)
    x_pca = pca.transform(x_test)
    x_error = x_pca[y_test != y_pred,:]
    if dataset == 'mnist':
        colors = plt.cm.tab10.colors
    else:
        colors = ['red','green','blue']
    plt.scatter(x_pca[:,0], x_pca[:,1], s=12, marker='x', c=y_test, cmap=ListedColormap(colors))
    plt.plot(x_error[:,0], x_error[:,1], 'ok', markersize=15, fillstyle='none')
    plt.title(title)
    plt.grid()
    plt.show()


# --------------------------------------- MAIN ---------------------------------------  
def main():
    # DATASET 

    K = [1,2,3,4,5,10,15,20,30,40,50]
    num_tests = 5
    mean_accuracies = []
    std_accuracies = []
    
    train_pca = {}
    test_pca  = {}
    pred_pca  = {}
    true_pca = {}

    DATASET = "wine"
    if(DATASET == 'mnist'):
        print("\n------------------MNIST-----------------\n")
        mnist = tf.keras.datasets.mnist
        (x_train_tot, y_train_tot), (x_test_tot, y_test_tot) = mnist.load_data()
        class_labels = np.unique(y_train_tot)        
        classes = len(class_labels)

    else:
        print("\n------------------WINE-----------------\n")
        wine = load_wine()
        X, y = wine.data, wine.target
        #print(wine.metadata) 
        #print(wine.variables) 

        # split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        class_labels = np.unique(y_train)        
        classes = len(class_labels)

        # scaling
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)


    # PREPROCESSING and DATA CLEANING | NOT needed in our assignment because we are imputing a 'toy problem'
        
    # accuracies for classes
    class_accuracies = {c: {k_val: [] for k_val in K} for c in class_labels}

    # CLASSIFIER
    for k in K:
        print(f"----- for k = {k} -----")
        if k % classes == 0:
            print(f"In this case k is divisible by the number of classes ({classes})")
        k_accuracy = []

        model: Knn =  Knn(n_neighbors=k)

        # subset
        if DATASET == 'mnist':
            train_size = 10000
            test_size = 2000
            train_subsets = []
            test_subsets = []  
            num_train_subsets = 6
            num_test_subsets = 5
            for _ in range(num_train_subsets):
                x_train, y_train, _, _ = mnist_subsets(x_train_tot, y_train_tot, x_test_tot, y_test_tot, train_size=train_size, test_size=1)
                train_subsets.append((x_train, y_train))
            for _ in range(num_test_subsets):
                _, _, x_test, y_test = mnist_subsets(x_train_tot, y_train_tot, x_test_tot, y_test_tot, train_size=1, test_size=test_size)
                test_subsets.append((x_test, y_test))

        for n in range(num_tests):
            if DATASET == 'mnist':
                i = np.random.choice(num_train_subsets)
                j = np.random.choice(num_test_subsets)
                x_train, y_train = train_subsets[i]
                x_test, y_test   = test_subsets[j]     
                print(f"--- Subset TRAIN = {i}, substet TEST {j} ---")
                

            # training
            model.fit(x_train, y_train)

            # Predicting 
            y_pred = model.predict(x_test)
            '''print(f"y_test type: {type(y_test)}")
            print(f"y_pred type: {type(y_pred)}")
            print(f"y_test[:5]: {y_test[:5]}")
            print(f"y_pred[:5]: {y_pred[:5]}")'''

            # Testing
            accuracy = model.test(y_test, y_pred)
            k_accuracy .append(accuracy)
            print(f"Test {n+1}: accuracy = {accuracy:.4f}")

            # accuracies fro classes (comparison one-all)
            for c in class_labels:
                acc_c = accuracy_score(y_test == c, y_pred == c)
                class_accuracies[c][k].append(acc_c)

            # save for pca - take the first test
            if n == 0:
                train_pca[k] = x_train
                test_pca[k]  = x_test
                pred_pca[k]  = y_pred
                true_pca[k]  = y_test

        mean_accuracies.append(np.mean(k_accuracy))
        std_accuracies.append(np.std(k_accuracy))
        
                    
    # DISPLAY 
    # summary
    print(f"\n===== {DATASET}: summary =====")
    print(f"{'k':>5} | {'mean_accuracy':>14} | {'std_accuracy':>12}")
    print("-" * 40)
    for k, mean_acc, std_acc in zip(K, mean_accuracies, std_accuracies):
        print(f"{k:5d} | {mean_acc:14.4f} | {std_acc:12.4f}")
    print()

    k_overfitting  = min(K)                        
    k_best         = K[np.argmax(mean_accuracies)]
    k_underfitting = max(K)                     

    #overfitting
    pca_plot(DATASET, train_pca[k_overfitting], test_pca[k_overfitting], true_pca[k_overfitting], pred_pca[k_overfitting], f'OVERFITTING - k={k_overfitting}')

    # best
    pca_plot(DATASET, train_pca[k_best], test_pca[k_best], true_pca[k_best], pred_pca[k_best], f'BEST - k={k_best}')

    # underfitting
    pca_plot(DATASET, train_pca[k_underfitting], test_pca[k_underfitting], true_pca[k_underfitting], pred_pca[k_underfitting], f'UNDERFITTING - k={k_underfitting}')

    # comparison for classes
    for c in class_labels:
        mean_each_k = [np.mean(class_accuracies[c][k_val]) for k_val in K]
        std_each_k  = [np.std(class_accuracies[c][k_val])  for k_val in K]

        plt.plot(K, [100 * m for m in mean_each_k], label=f'Class {c}', marker='o', markersize=5, linewidth=2, alpha=0.4)
        plt.fill_between(K, [100*(m - s) for m, s in zip(mean_each_k, std_each_k)], [100*(m + s) for m, s in zip(mean_each_k, std_each_k)], alpha=0.15)

    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy per class (%)')
    plt.title('Accuracy for class')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.15)
    plt.xticks(K)
    plt.show()

if __name__ == '__main__':  
    main()