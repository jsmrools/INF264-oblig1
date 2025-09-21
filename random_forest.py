import numpy as np
import random
from decision_tree import DecisionTree
from decision_tree import most_common

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features

    def fit(self, X: np.ndarray, y: np.ndarray): #training the forest     
        self.trees = [] #this will have all our trained decition trees
        samples = X.shape[0] #tells us how many samples we have, our case 2000

        #Bootstrapping: -> to get 20 different trees (with same data length = 2000) from the same dataset.
        for i in range(self.n_estimators): #n_estimators = 20, then 20 trees
            random_indices = np.random.choice(samples,samples,replace=True)   #important to use indices so we can make sure we keep track of both features and labels
            X_sample, y_sample = X[random_indices], y[random_indices]
            tree = DecisionTree(max_depth=self.max_depth,criterion=self.criterion, max_features = self.max_features) #passes max_features to DT
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)


    def predict(self, X: np.ndarray) -> np.ndarray:
        total_tree_predictions = []
        for tree in self.trees: #we look at the trained trees ojects from fit
            prediction = tree.predict(X) #for each tree, we will take that tree and predict on all the rows of X. when we call it predict(X_val) we use the unseen data to se how well the model calculates new data
            total_tree_predictions.append(prediction)                               
        total_tree_predictions = np.array(total_tree_predictions) #convert it into a Numpy arrar after the loop 
        
        transposed_tot_tree_pred = total_tree_predictions.T #makes sures the samples becomes the rows and the tree become the columns. so that we can inspect the majorities inside the samples
        final_prediction_array = []
        for row in transposed_tot_tree_pred:
            majority_value = most_common(row)
            final_prediction_array.append(majority_value)
        return np.array(final_prediction_array)



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features="sqrt"  #Makes a random forest with 20 trees.
    )
    rf.fit(X_train, y_train) #trains the random forest

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
