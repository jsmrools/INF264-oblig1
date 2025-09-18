import numpy as np
from typing import Self

"""
This is a suggested template and you do not need to follow it. You can change any part of it to fit your needs.
There are some helper functions that might be useful to implement first.
At the end there is some test code that you can use to test your implementation on synthetic data by running this file.
"""

def count(y: np.ndarray) -> np.ndarray:
    unique_values = np.unique(y) #converts to an np as an new array with the unique elements from y

    value_probabilities = []
    numbers_in_array = len(y)

    for value in unique_values:
        number_of_occurrences= np.sum(y == value) #creates a boolean numpy array (false,true,true) and adds up the true values.
                                                    #we need np. as it is much faster than just sum (importent for large datasets)
        value_probability = number_of_occurrences/numbers_in_array
        value_probabilities.append(value_probability)

    return np.array(value_probabilities) #converting the list into a numpy array 
 



def gini_index(y: np.ndarray) -> float:
    probabilities = count(y)

    squared_probabilities = []
    for probability in probabilities:
        squared_probabilities.append(probability ** 2)
    
    gini = 1 - sum(squared_probabilities)
    return float(gini)


def entropy(y: np.ndarray) -> float:
    probabilities = count(y)

    multiplied_and_log_probalilities = []

    for probability in probabilities:
        if probability >0: 
            multiplied_and_log_probalilities.append(probability * np.log2(probability))
        else:
            multiplied_and_log_probalilities.append(0) #we need to have an catch all to the cases where the probability is 0, 
                                                        #because computing log(0) would cause an error
    
    entropy = -sum(multiplied_and_log_probalilities)
    return float(entropy)


def split(x: np.ndarray, value: float) -> np.ndarray: #returns an array of booleans [True,True,False]
    mask = x <= value #filters to a set of booleans, called mask because it is (masking) parts of the data 
    return mask        #If its true then it goes left, if false right
    

def most_common(y: np.ndarray) -> int:
    unique_values = np.unique(y)

    total_value_count = 0
    highest_value = None


    for value in unique_values:
        value_count = np.sum(y == value)

        if value_count > total_value_count:
            total_value_count = value_count
            highest_value = value

    return int(highest_value)


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0, # which column of x this node splits on
        threshold: float = 0.0, # cutoff value for splitting
        left: int | Self | None = None, #after split data points go left
        right: int | Self | None = None, #data points go right
        value: int | None = None, # if value is a leaf node (stores the predicted class)
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        #internal node may look like Node(feature=2, threshold=0.5, left=..., right=...)


    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        # helper to check if node is leaf node
        # mightlook like Node(value = 1)
        return self.value is not None
    
    def to_string(self, depth=0, prefix=""):
        """
        Return a string representation of the tree rooted at this node.
        Uses indentation and ASCII branches for visualization.
        """
        indent = "    " * depth
        if self.is_leaf():
            return f"{indent}{prefix}Leaf → predict {self.value}"
        else:
            s = f"{indent}{prefix}Feature {self.feature} <= {self.threshold}\n"
            s += self.left.to_string(depth + 1, "├── ") + "\n"
            s += self.right.to_string(depth + 1, "└── ")
            return s


class DecisionTree:
    # controller class that builds the decision tree
    def __init__(self, max_depth: int | None = None, criterion: str = "entropy", max_features=None) -> None:
        if criterion not in ("entropy","gini"):
            raise ValueError("the criterion must be 'entropy' or 'gini'") #makes sure the users put in a valid criterion
        self.root = None #store the top Node of the entire tree once its trained
        self.criterion = criterion # which impurity measure to use (entropy or gini)
        self.max_depth = max_depth # limits how deep the tree can grow (prevents overfitting)
        self.max_features = max_features
        
        if self.criterion == "entropy":
            self.impurityf = entropy
        else:
            self.impurityf = gini_index
       

    def find_threshold(self, x_column: np.ndarray, y: np.ndarray):
        """
    Given a single feature column and labels y,
    find the threshold that gives the best information gain.
    Returns (best_threshold, best_gain).
    #gain = impurity(before split) - weighted impurity(after split)
    """
        currentnodeimp = self.impurityf(y)
        values = np.unique(x_column)
        best_column_gain = -1 # for testing of best column gain
        best_column_threshold = None
        for threshold in values:
            mask = split(x_column, threshold) #creates a split based on a threshold made from values in column
            left_y, right_y = y[mask], y[~mask] # splits the current column into "left and right labels" based on the mask for x_column
            if len(left_y) == 0 or len(right_y) == 0: #hopping over useless splits
                continue
            #testing the threshold "if i test here how mixed are the child nodes"
            n, n_left, n_right = len(y), len(left_y), len(right_y)
            child_impurity = (n_left/n) * self.impurityf(left_y) + (n_right/n) * self.impurityf(right_y) #weighted mean impurity of child nodes
            gain_for_split = currentnodeimp - child_impurity
            if gain_for_split > best_column_gain: #checks if this is the best gain split for the column
                best_column_gain = gain_for_split
                best_column_threshold = threshold
        return best_column_threshold, best_column_gain #returns the best column threshold and gain
        
    def find_best_feature(self, X: np.ndarray, y: np.ndarray):
        """
    Loop over all features, call find_threshold for each,
    and return the best (feature, threshold).
    """
        all_features = X.shape[1]

        #______CHOOSE # OF FEATURES_____ ##FOR RANDOM FOREST
        if self.max_features == "sqrt":
            number_of_feat = int(np.sqrt(all_features))
        elif self.max_features == "log2":
            number_of_feat = int(np.log2(all_features))
        elif self.max_features is None:
            number_of_feat = all_features
        else:
            number_of_feat = int(self.max_features) #this is purely just extra, if there was a need to decide this not based on log or sqrt
        # if there is a data set with one feature it will crash the fix:
        number_of_feat = max(1,number_of_feat)
        number_of_feat = min(number_of_feat, all_features) #making sure to reduce error if person passes a large numer larger than num og feat


        # randomly select features w/out replacement, if the number of features doesnt change based on max_feat then it loops through em all
        if number_of_feat == all_features:
            rand_feat_int = np.arange(all_features)
        else: #creates a random sample of the features for each split
            rand_feat_int = np.random.choice(all_features, number_of_feat, replace = False)

        #search for the best split among features
        best_gain = -1 #for overall testing of best feature
        best_feature = None
        best_threshold = None
        for feature in rand_feat_int: #loops over the array of choosen features
            x_column = X[:, feature]
            threshold, gain = self.find_threshold(x_column, y)
            if  gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth = 0): #building the tree is recursive whereas fit is used only once so build tree def is defined
    #step 1, stopping conditions
        if len(np.unique(y)) == 1: #check if all labels are the same
            return Node(value = y[0]) #node has label of the only label left
        
        if len(np.unique(X, axis=0)) == 1: # unique(X, axis = 0) tells NumPy to look at rows that are unique not elements
            return Node(value=most_common(y)) #returns label of the most common label in the node currently

        if self.max_depth is not None and depth >= self.max_depth: # if max depth defined and the current depth is more or equal to max depth
            return Node(value=most_common(y)) #returns label of the most common label in the node currently
        
    #step 2, choose best split to recursively split the dataset so the labels in each subset become more “pure”
        # best feature/threshold: column in dataset that does the best job of separating the labels; use of best split function
        best_feature, best_threshold = self.find_best_feature(X, y) #finding best feature uses find threshold and returns both threshold and feature

        if best_feature is None or best_threshold is None: #checks if None is passed and if so value is most comon y
            return Node(value=most_common(y))
        
        # create node
        node = Node(feature = best_feature, threshold= best_threshold)
        # partition data 
        mask = split(X[:,best_feature], best_threshold)
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]
        # recurse
        node.left = self.build_tree(left_X, left_y, depth +1)
        node.right = self.build_tree(right_X, right_y, depth +1)
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray,): #recursive builder
        self.root = self.build_tree(X, y, depth = 0)
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y but is not recursive because called once.
        """
        # ID3 algorithm 
            #find the best split at the root
            #create a Node
            #recursivly split left and right child datasets until stopping conditions(max depth, pure labels)
            # store final root node in self.root
        #BUILDING THE ACTUAL TREE BASED ON TEST DATA

        # create a Node(feature= .., threshold == )
            #recursivly assigns Node.left and Node.right by calling itself on subsets of the data
    
    def traverse(self, node: Node, row: np.ndarray):
        if node.is_leaf():
            return node.value
        if row[node.feature] <= node.threshold: #defining which way to go

            return self.traverse(node.left, row)
        else:
            return self.traverse(node.right, row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        #Used after training
        #same as fit keep it clean use a helper function to encapsulate recusion.
        #for each row X start at self.root and follow the branches, check thresholds and features
        #until you land in a leaf node then return the leafs value
        results = []
        start_node = self.root #for efficiency 
        for row in X: # defines a loop that takes in a object in data X(Test/validation)
            label = self.traverse(start_node, row)
            results.append(label)
        return np.array(results)
        


if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
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

    # looking at the data:

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="entropy")
    
    rf.fit(X_train, y_train)


    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(rf.root.to_string())
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")