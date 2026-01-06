# main.py 
#   - before running, set the desired TASK and CLASSES variables
#   - to run all tasks sequentially, set RUN_ALL = True, otherwise set TASK to 1, 2, or 3


import numpy as np
import tensorflow as tf

from task1 import prepare_data
from task2 import run_task2
from task3 import run_task3

def main():
    
    #lab settings
    RUN_ALL = False 
    TASK = 2 #1 (data prep), 2 (autoencoder 2D), 3 (inpainting)
    CLASSES = (5, 7) # for tasks 1 and 2
    SEED = 42 #riproducibility

    #riproducibility
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    #run all tasks cosequently
    if RUN_ALL:
        print(f"\nTASK 1 - Preparing data for classes {CLASSES}")
        X_train, X_test, y_train, y_test = prepare_data(classes=CLASSES)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        print(f"--------------------------------")
        print(f"\nTASK 2 - Autoencoder embedding for classes {CLASSES}")
        run_task2(X_train=X_train, X_test=X_test, y_test=y_test, classes=CLASSES)

        print(f"--------------------------------")
        print("\nTASK 3 - Masked reconstruction (all MNIST classes)")
        X_train, X_test, _, _ = prepare_data(classes=None)
        run_task3(
            X_train=X_train,
            X_test=X_test,
            mode="mix",
            severities=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            epochs=20,
            batch_size=256
        )

    #task1: data preparation
    elif TASK == 1:
        print(f"--------------------------------")
        print(f"\nTASK 1 - Preparing data for classes {CLASSES}")
        
        X_train, X_test, y_train, y_test = prepare_data(classes=CLASSES)

        #debug
        print(f"  Train set shape: {X_train.shape} | y_train: {y_train.shape}")
        print(f"  Test set shape:  {X_test.shape}  | y_test:  {y_test.shape}")


    #task2: autoencoder embedding
    elif TASK == 2:
        print(f"--------------------------------")
        print(f"\nTASK 2 - Autoencoder embedding for classes {CLASSES}")

        X_train, X_test, y_train, y_test = prepare_data(classes=CLASSES)
        run_task2(X_train=X_train, X_test=X_test, y_test=y_test, classes=CLASSES)

        ''' RESULTS
            classes (5, 7):
                the classes are well separated, so the encoder is able to learn features that distinguish the two digits

            classes (3, 5):
                as expected the two classes are mixed since the digit are similar.

            So we can conclude that the autoencoder is able to learn useful features for digits that are different, 
            but struggles when the digits are similar.
        '''


    #task3: masked reconstruction
    elif TASK == 3:
        print(f"--------------------------------")
        print("\nTASK 3 - Masked reconstruction (all MNIST classes)")

        X_train, X_test, _, _ = prepare_data(classes=None)  #all classes
        run_task3(
            X_train=X_train,
            X_test=X_test,
            mode="mix",
            severities=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            epochs=20,
            batch_size=256
        )
        
        ''' RESULTS
            Severity   Test MSE   Quality   
            --------------------------------
            0.10       0.0110     Excellent 
            0.20       0.0168     Excellent 
            0.30       0.0227     Good      
            0.40       0.0298     Good      
            0.50       0.0376     Good      
            0.60       0.0451     Okay      
            0.70       0.0539     Okay       

            As the severity of the masking increases, the test MSE also increases, indicating that the reconstruction quality decreases.
            Until severity as 0.5 the model performs reasonably well (Excellent to Good) so the image is still faithful to the original.
            However, at severity 0.6, the quality drops to Okay, showing that the reconstruction is partially successful but no longer reliable
        '''

    else:
        raise ValueError("Invalid TASK number: choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
