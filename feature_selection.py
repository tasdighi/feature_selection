import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

class FeatureSelection:

        def sfs_with_lda(self, data, fixed_features, min_feature, max_feature):
                # sfs: sequential feature selection 
                # this method belongs to wrapper methods of feature selection strategies
                # looking for best subsets of features via LDA model based on model accuracy
                X = data.drop('target',1)
                y = data['target']
                y_df = pd.DataFrame(y)

                #forward search via LDA
                mlxtend_sfs = SFS(LDA(),
                        k_features=(min_feature, max_feature),
                        forward=True,
                        floating=False,
                        scoring = 'accuracy',
                        fixed_features=fixed_features,
                        cv = 0)

                sfs = mlxtend_sfs.fit(X,y_df)
                sfs_result = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
                print('Summary: ', sfs_result.sort_values('avg_score', ascending = False))

        def fisher_score(self, X, target):
                # this method calculate fisher score and
                # can be used as filter method of feature selection strategies

                #number of dimentions
                dim = len(X.columns.tolist()) 

                #Compute the mean vector as mu and the mean vector per class as mu_k on each dimention
                mu = np.mean(X,axis=0).values.reshape(dim,1)           
                mu_k = np.zeros((len(np.unique(target)), dim))
                within_class_pow = np.zeros((len(np.unique(target)), 1))
                for i,item in enumerate(np.unique(target)):
                        mu_k[i] = np.mean(X.where(target==item),axis=0)
                        a = np.array(X.where(target==item).dropna().values-mu_k[i,:])
                        within_class_pow[i] = sum(sum(a**2))

                #Compute the within and between distances
                mu_k = np.array(mu_k).T
                between_class_pow = (mu_k - mu) **2
                between_class = sum(sum(between_class_pow))
                within_class = sum(within_class_pow)
                
                #Compute the fisher score
                score = between_class / within_class
                return score

        def sfs_with_fisher(self, data, fixed_features):
                # this method is a combination of wrapper and filter methods of feature selection strategies
                #looking greedy for best subsets of features based on subset fisher score.
                #all subsets include fixed_features
                 
                y = data['target']
                X = data.drop('target',1)
                
                initial_features = X.columns.tolist()
                best_features = fixed_features
                score = dict()
                
                while (len(initial_features)>0):
                        remaining_features = list(set(initial_features)-set(best_features))
                        new_score = pd.Series(index=remaining_features)
                        for new_column in remaining_features:
                                new_score[new_column] = self.fisher_score(X[best_features+[new_column]],y)
                        max_score = new_score.max()       
                        if(not(np.isnan(max_score))):
                                best_features.append(new_score.idxmax())
                                score[str(best_features)] = max_score
                        else:
                                break
                return score
        
        def all_subsets(self, category_dict):
                #category_dict format: {'cat_name': {cat_features': value, 'cat_fixed':value, 'cat_min': value, 'cat_max':value}}
                #find all combinations of cat_features that include cat_fixed features
                #also minimum/ maximum number of category members is cat_min/cat_max
                
                all_cat_valid_set = []
                for value in category_dict.values():
                        #exclude fixed_feature to improve performance
                        cat_fixed_features = value['cat_fixed'] if 'cat_fixed' in value.keys() else []
                        cat_feature_set = list(set(value['cat_features'])-set(cat_fixed_features))

                        #find all combination of feature_set exclude fixed_feature
                        cat_sets = list(map(list, list(chain.from_iterable(list(combinations(cat_feature_set, r) for r in range(len(cat_feature_set)+1))))))
                        
                        #include fixed_feature in each combination and set min/max number of category member
                        cat_min = value['cat_min'] if 'cat_min' in value.keys() else 1
                        cat_max = value['cat_max'] if 'cat_max' in value.keys() else len(value['cat_features'])
                        cat_valid_sets = []
                        for each_set in cat_sets:
                                each_set.extend(cat_fixed_features)
                                if len(each_set)>=cat_min and len(each_set)<=cat_max:
                                        cat_valid_sets.append(each_set)
                        all_cat_valid_set.append(cat_valid_sets)

                #extract product of feature set members
                feature_set = reduce(lambda x, y: map(lambda p: p[0]+p[1], product(x, y)), all_cat_valid_set[:])

                return feature_set
        
        def compelete_search_with_fisher(self, data, category_dict):
                #find all limited subsets of features by count or any fixed feature, mentioned in category_dict
                #caclulate fisher score for all sub sets

                X = data.drop("target", 1)       
                y = data['target']               
                subsets = self.all_subsets(category_dict)
                subsets_score=dict()
                for subset in subsets:
                        subsets_score[str(subset)] = self.fisher_score(X[subset],y)
                        
                return subsets_score


