from collections import defaultdict
import numpy as np
import pandas as pd

# BAYER CLASSIFIER

# FIT FUNCTIONS
# Calculate the priors for eash class: P(c) = count_c / n 
def calculate_class_priors(y):
    y = pd.Series(y)
    class_counts = y.value_counts()
    n = len(y)
    class_priors = class_counts.div(n).to_dict()
        #print("priors: ",class_priors) 
    return class_priors


# Calculate the conditional prob for each feature and class: P(t | t=yi)
def calculate_feature_class_likelihood(x, y, levels, alpha=1):
    X = pd.DataFrame(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    all_classes = pd.Index(sorted(y.unique())) 
        #print(all_classes) 
    feature_class_likelihood = {}

    for feature in X.columns:
        all_levels = pd.Index(levels[feature])
            #print(all_levels) 

        #pandas.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)  
        #crosstab -> return a DataFrame
        counts = pd.crosstab(y, X[feature])
        counts = counts.reindex(index=all_classes, columns=all_levels, fill_value=0)
        v = len(all_levels)
        den = counts.sum(axis=1)+alpha*v
        likelihood = (counts+alpha).div(den, axis=0)
        feature_class_likelihood[feature] = likelihood

        ''' check
        print(f"\n[Feature: {feature}]")
        print("class x feature level:\n", counts)
        print("P(t | t=yi):\n", likelihood)'''

    return feature_class_likelihood


#----------------------------------------------------------------------------------------------------------------------------#

# CLASS

# Discrete Naive Bayes model
class Nbayes:
    def __init__(self, unknown='error' ,alpha=1):
         self.trained = False
         self.class_priors = {} 
         self.log_class_priors = {}
         self.feature_class_likelihood = {}
         self.feature_class_loglikelihood = {}
         self.features_ = []        
         self.classes_ = []      
         self.unknown = unknown  
         self.levels = {}
         self.alpha = alpha
         '''
         ERROR:
         - error -> stop and warning
         - discard -> jump to the next line
         '''

    # training model
    def fit(self, x, y, levels=None):
        X = pd.DataFrame(x)
        y = pd.Series(y)

        if levels is None:
            levels = {col: sorted(x[col].dropna().unique().tolist()) for col in X.columns}

        self.levels = levels
        self.features_ = list(X.columns) 
        self.classes_  = list(y.unique())

        self.class_priors = calculate_class_priors(y)
        self.log_class_priors = {c: np.log(p) for c, p in self.class_priors.items()}
        
        self.feature_class_likelihood = calculate_feature_class_likelihood(x, y, levels=levels, alpha=self.alpha)  
        self.feature_class_loglikelihood = {f: np.log(df.astype(float)) for f, df in self.feature_class_likelihood.items()}
        self.trained = True

        ''''
        for f, tab in self.feature_class_likelihood.items():
            print(f"\n[Feature: {f}]")
            print(tab)
        every row = 1 -> OK
        '''

        return self


    #P(t|y) = (P(y|t = yi)*P(t=ti))/P(y)
    def predict(self, x):
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")

        X = pd.DataFrame(x)[self.features_]  #same order of the train one
        y_pred = []

            #print(f"Features: {self.features_}")
            #print(f"Class priors: {self.class_priors}\n")

        # feature -> value
        for i in range(len(x)):
            x_i = X.iloc[i]
                #print(f"\n row {i+1}:",x_i.to_dict())

            # unseen level
            unseen = []
            for f in self.features_:
                v = x_i[f]
                if pd.isna(v):
                    unseen.append((f, v))
                elif v not in self.levels[f]:
                    unseen.append((f, v))
            if unseen:
                if self.unknown == 'error':
                    raise ValueError(f"Unseen level at row {i}: {unseen}")
                elif self.unknown == 'discard':
                    y_pred.append(None)
                    continue
                raise ValueError(f"Unknown handling mode '{self.unknown}'")

            # log P(c) + sum_f log P(x_f=v | c)
            best_class = None
            best_score = -np.inf
            for c in self.classes_:
                score = self.log_class_priors[c]
                
                    #print(f"  Classe '{c}': prior = {prior:.4f}")
                for f in self.features_:
                    v = x_i[f]
                    if pd.isna(v):
                        score = -np.inf
                        break
                    probs_df = self.feature_class_loglikelihood[f]
                    score += float(probs_df.loc[c, v])

                    #print(f"  → Score totale per classe '{c}': {score:.6f}")
                if score > best_score:
                    best_score = score
                    best_class = c
                #print(f"PREDICY: {best_class} (score={best_score:.6f})")
            y_pred.append(best_class)
        return np.array(y_pred)

     #testing
    def test(self, X_test, y_test):
        if not self.trained:
            raise ValueError("Model not trained. Call fit() first.")
        y_pred = self.predict(X_test)

        return (np.array(y_test) == np.array(y_pred)).sum() / len(y_test)