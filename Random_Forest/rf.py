
import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False,min_samples_leaf=3,max_features=0.3):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.trees=[]

    def compute_oob_score(self,X,y):

        oob_index_values={key:[] for key in range(len(X))}
        oob_index_preds={}
        
        if isinstance(self, RandomForestRegressor621):

            for tree in self.trees:
            
           

                oob_predictions_leaves=tree.predict(X[tree.oob_index])
                oob_predictions=[leaf.y_vals for leaf in oob_predictions_leaves]

                for i in range(len(oob_predictions)):
                
                    index=tree.oob_index[i]

                    oob_index_values[index].append(oob_predictions[i])

        #loss = self.trees[0].loss
        #loss(y)
            for i in range(len(X)):
                try:
                    oob_index_values[i]=np.concatenate(oob_index_values[i])
                except:
                    continue
                
                oob_index_preds[i]=np.mean(oob_index_values[i])

            #Have to deal with missing values
            
            oob_predictions=list(oob_index_preds.values())
            oob_predictions=np.array(oob_predictions)
            oob_predictions=oob_predictions[np.where(~np.isnan(oob_predictions))[0]]
            y_filtered=y[np.where(~np.isnan(oob_predictions))[0]]
        #arr[np.where(~np.isnan(test_lis))[0]]


            return r2_score(y_filtered,oob_predictions)
        
        
        elif isinstance(self, RandomForestClassifier621):
            
                for tree in self.trees:
        
                    oob_predictions_leaves=tree.predict(X[tree.oob_index])
                    oob_predictions=[leaf.y_vals for leaf in oob_predictions_leaves]

                    for i in range(len(oob_predictions)):
                
                        index=tree.oob_index[i]

                        oob_index_values[index].append(oob_predictions[i])



        #loss = self.trees[0].loss
        #loss(y)
                for i in range(len(X)):
                    #stats.mode(rf.oob_score_[0])[0][0]
                    try:
                        oob_index_values[i]=np.concatenate(oob_index_values[i])
                    except:
                        continue
                    oob_index_preds[i]=stats.mode(oob_index_values[i])[0]

            #Have to deal with missing values
            
                oob_predictions=list(oob_index_preds.values())
                oob_predictions=np.concatenate(oob_predictions)
                #oob_predictions=[oob_predictions[k][0] for k in range(len(X_train))]
                oob_predictions=np.array(oob_predictions)
                oob_predictions=oob_predictions[np.where(~np.isnan(oob_predictions))[0]]
                y_filtered=y[np.where(~np.isnan(oob_predictions))[0]]
        #arr[np.where(~np.isnan(test_lis))[0]]
                #return oob_index_values
                #return oob_predictions
                return accuracy_score(y_filtered,oob_predictions)
    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """

        oob_index_dict={}
        trees_dict={}

        for i in range(self.n_estimators):
            n = len(y)
            idx = np.random.randint(0, n, size = n)
            X_train = X[idx]
            y_train = y[idx]
            oob_index=np.array([i for i in range(len(X)) if i not in idx])
            #oob_index_dict[i]=oob_index
            if isinstance(self, RandomForestRegressor621):
                tree=RegressionTree621(self.min_samples_leaf, self.max_features, oob_index)
                tree.fit(X_train,y_train)
            #trees_dict[i]=tree
                self.trees.append(tree)
                if self.oob_score==True:
                    self.oob_score_=self.compute_oob_score(X,y)

            elif isinstance(self, RandomForestClassifier621):
                tree=ClassifierTree621(self.min_samples_leaf, self.max_features, oob_index)
                tree.fit(X_train,y_train)
                self.trees.append(tree)
                if self.oob_score==True:
                    self.oob_score_=self.compute_oob_score(X,y)
                #self.oob_score_=self.compute_oob_score(self,X,y)

class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score, min_samples_leaf=min_samples_leaf, max_features=max_features)
        #self.trees = ...

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        
        prediction_leaves=[]
        weighted_predictions=[]
        count_leaves=[]

        for tree in self.trees:

            leaves=tree.predict(X_test)
            prediction_leaves.append(leaves)



        for i in range(self.n_estimators):

            weighted_predictions.append([leaf.prediction*leaf.n for leaf in prediction_leaves[i]])
            count_leaves.append([leaf.n for leaf in prediction_leaves[i]])

        weighted_arr=np.array(weighted_predictions)
        weighted_pred_sum=np.sum(weighted_arr,axis=0)
        count_leaves_arr=np.array(count_leaves)
        leaves_sum_count=np.sum(count_leaves_arr,axis=0)

        final_prediction_array=weighted_pred_sum/leaves_sum_count

        return final_prediction_array
    
      
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        #self.trees = ...

    def predict(self, X_test) -> np.ndarray:
        
        prediction_leaves=[]
        #weighted_predictions=[]
        #count_leaves=[]
        list_y_vals=[]
        final_list=[]

        for tree in self.trees:

            leaves=tree.predict(X_test)
            prediction_leaves.append(leaves)
            
        

        for i in range(self.n_estimators):

            list_y_vals.append([leaf.y_vals for leaf in prediction_leaves[i]])
            
        for j in range(len(X_test)):


            final_list.append(np.concatenate([list_y_vals[i][j] for i in range(self.n_estimators)]))
            
           
        return np.array([stats.mode(final_list[i])[0][0] for i in range(len(X_test))])
        
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
        

