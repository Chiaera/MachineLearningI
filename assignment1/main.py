from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from Nbayes import Nbayes

def main():
    # IMPORT DATA SET
    main_dir = Path(__file__).resolve().parent
    data_path = main_dir / "data"
    
    DATASET = "breast"  
    if DATASET == "weather":
        print("--- WEATHER ---")
        weather_path = data_path / "weatherData/weatherData.txt"
        df = pd.read_csv(weather_path, delim_whitespace=True, header=0)
        #print(df.head()) -> table OK
        #print(df.shape) -> shape OK 
    else:
        print("--- BREAST CANCER ---")
        breast_path = data_path / "breastCancer/breast-cancer.data"
        columns = ["Class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "breast", "breast-quad", "irradiat"]
        df = pd.read_csv(breast_path, names=columns, header=None)
        '''
            print("Righe iniziali:", len(df))
            print("Duplicati (righe identiche):", df.duplicated().sum())
            print("NaN in Class:", df['Class'].isna().sum())
            print(df.head()) 
            print(df.shape)
        '''


    # DATA CLEANING
    if DATASET == "weather":
        target = 'Play'
    else:
        target = 'Class'
    df.replace('?', np.nan, inplace=True)
    df.drop_duplicates(inplace=True)   
        #print(df) -> replacement OK
    df.dropna(subset=target, inplace=True)
        #print(df) -> row deleted OK



    # FEATURES (all categorical) and TARGET
    feature= [c for c in df.columns if c != target]
    X = df[feature]
    y = df[target]

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=42)


    # train LEVEL 
    levels = {col: sorted(X_train[col].dropna().unique().tolist()) for col in feature}
        #print(levels)

    # MISSING categorical data
    miss_train = (X_train.isna().sum() / len(X_train)) * 100
    col_miss_train = miss_train[miss_train > 0].index
        #print(miss_train[col_miss_train]) -> missing data for a colomn OK

    if len(col_miss_train) > 0:
        if all(miss_train[col_miss_train]<=15):
            imputer = SimpleImputer(strategy='most_frequent')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature, index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=feature, index=X_test.index)
        else:
            X_train = X_train.dropna(subset=col_miss_train)
            y_train = y_train.loc[X_train.index]
            X_test = X_test.dropna(subset=col_miss_train)
            y_test = y_test.loc[X_test.index]
        '''
            print("TRAIN set: ", X_train.shape)
            print("TEST set:", X_test.shape)
            print("y_train dist:\n", y_train.value_counts(normalize=True))
            print("y_test  dist:\n", y_test.value_counts(normalize=True))
        '''


    # CHECK
    if (X_train.shape[1] != X_test.shape[1]):
        raise ValueError("Dimensional error: X_train and X_test must have same number of columns.")
    if y_train is None or y_test is None:
        raise ValueError("Dimensional error", "Issue with of y test set's columns.")


    #--------------------------------------------------------------------------------------------------------------------------------------------------#

    # BUILD  BAYES CLASSIFIER
    # fit
    model: Nbayes = Nbayes(unknown='discard', alpha=1)
    model.fit(X_train, y_train, levels=levels)

    # predict
    y_pred = model.predict(X_test)

    #test
    accuracy = model.test(X_test, y_test)
    print(f"Accuracy: {accuracy:.3f}")

    n_discard = np.sum([p is None for p in y_pred])
    discard = [i for i, p in enumerate(y_pred) if p is None]
    if n_discard > 0:
        print(f"Discard: {n_discard} su {len(y_pred)}")
        for i in discard:
            print(f"Row {i}: {X_test.iloc[i].to_dict()}")
        


if __name__ == '__main__':  
    main()
