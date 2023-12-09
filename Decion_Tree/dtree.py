import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        
        if x_test[self.col]>=self.split:
            return self.rchild.predict(x_test)
        else:
            return self.lchild.predict(x_test)
        


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        

    def predict(self, x_test):
        return self.prediction
        ...
        
        


def gini(y):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    ...
    
    y=list(y)
    p_0=(y.count(0))/len(y)
    p_1=(y.count(1))/len(y)
    p_0_2=p_0**2
    p_1_2=p_1**2
    
    g=1-(p_0_2 + p_1_2)
    
    return g

    
    
    
def find_best_split(X, y,loss, min_samples_leaf):
    
    feature=-1
    split=-1
    min_loss=loss(y)
     
    for col in range(np.shape(X)[1]):
        split_values = np.random.choice(X[:,col], 11)
        for sv in split_values:
            y_left = y[X[:,col] < sv]
            y_right = y[X[:,col] >= sv]

            len1 = np.shape(y_left)[0]
            len2 = np.shape(y_right)[0]

            if (len1 < min_samples_leaf) or (len2 < min_samples_leaf):
                continue
                         
            else:
                los = ((len1 * loss(y_left)) + (len2 * loss(y_right)))/(len1 + len2)
                if los == 0:
                    return col,sv
                elif los < min_loss:
                    feature=col
                    split=sv
                    min_loss=los
    return feature,split
            
               
            #for split in range(np.min(X_t),np.max(X_t)):    
class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.var for regression or gini for classification
        
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressions predict the average y
        for observations in that leaf.
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classification or regression.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621.create_leaf() depending
        on the type of self.
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.
        (Make sure to call fit_() not fit() recursively.)
        """
        ...
        
        if np.shape(X)[0] <= self.min_samples_leaf or np.shape(np.unique(X, axis=0)[0])==1:
            return self.create_leaf(y)
        
        col,split=find_best_split(X,y,self.loss,self.min_samples_leaf)
        
        if col==-1:
            return self.create_leaf(y)
        else:
        #node=DecisionNode(col,split,fit_(self,X,y)
            y_reshape=np.reshape(y,(np.shape(y)[0],1)) #reshaping so that dimensions align
            Xy=np.hstack((X,y_reshape))
            
            lchild=self.fit_(Xy[Xy[:,col]<split][:,:-1],Xy[Xy[:,col]<split][:,-1])
            rchild=self.fit_(Xy[Xy[:,col]>=split][:,:-1],Xy[Xy[:,col]>=split][:,-1])
            
            tree=DecisionNode(col,split,lchild,rchild)
            
            return tree
                      

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        ...
        prediction_list=[]
        
        for x in X_test:
            prediction_list.append(self.root.predict(x))
        return np.array(prediction_list)
        
class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        ...
        predictions=self.predict(X_test)
        
        return r2_score(predictions,y_test)

    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        leaf=LeafNode(y,np.mean(y))
        
        return leaf
        


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        ...
        predictions=self.predict(X_test)
        
        return accuracy_score(predictions,y_test)
        
        

    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. Feel free to use scipy.stats to use the mode function.
        """
        ...
        
        leaf=LeafNode(y,stats.mode(y).mode[0])
        
        return leaf
        