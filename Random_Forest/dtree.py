import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

def gini(x):
    unique, counts = np.unique(x, return_counts=True)
    n = len(x)
    return 1 - np.sum( (counts / n)**2 )

def find_best_split(X, y, loss, min_samples_leaf,max_features):
    best = {'col':-1, 'split':-1, 'loss':loss(y)}
    for col in np.random.choice(len(X[0]),size = int(len(X[0]) * max_features), replace = True):
        candidates = np.random.choice(X[:, col], size = 11, replace = True)
        for sp in candidates:
            lhs = X[:, col] < sp
            rhs = X[:, col] >= sp
            yl = y[lhs]
            yr = y[rhs]
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
            l = (len(yl)*loss(yl) + len(yr)*loss(yr))/(len(y))
            if l == 0:
                return col, sp
            if l < best['loss']:
                best = {'col':col, 'split':sp, 'loss':l}
    return best['col'],best['split']

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)

    
    def leaf(self,x_test):
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        else:
            return self.rchild.predict(x_test)



        
class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y_vals=y

    def predict(self, x_test):
        # return prediction
        return self
        
class DecisionTree621:
    def __init__(self, min_samples_leaf=3, max_features=0.3, oob_index=[], loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var or gini
        self.max_features=max_features
        self.oob_index=oob_index


    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for observations in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)
        
    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf or len(np.unique(X))==1:
            return self.create_leaf(y)
        else:
            # find best split
            col, split = find_best_split(X, y, self.loss, self.min_samples_leaf,self.max_features)
            # return if no better split
            if col == -1:
                return self.create_leaf(y)
            lhs = X[:, col] < split
            rhs = X[:, col] >= split
            lchild = self.fit_(X[lhs], y[lhs])
            rchild = self.fit_(X[rhs], y[rhs])
            return DecisionNode(col, split, lchild, rchild)
        
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        return np.array([self.root.predict(X_test[ri,:]) for ri in range(len(X_test))])
        
class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=3, max_features=0.3, oob_index=[]):
        super().__init__(min_samples_leaf, max_features, oob_index, loss=np.var)
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, prediction = np.mean(y))
                        
class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=3, max_features=0.3, oob_index=[]):
        super().__init__(min_samples_leaf, max_features, oob_index, loss=gini)
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to get the mode.
        """
        return LeafNode(y, prediction = stats.mode(y)[0])