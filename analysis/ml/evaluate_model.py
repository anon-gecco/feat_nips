import sys
import itertools
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.pipeline import Pipeline,make_pipeline
from metrics import balanced_accuracy_score
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
from utils import feature_importance , roc
import pdb
import numpy as np

def evaluate_model(dataset, save_file, random_state, clf, clf_name, hyper_params, classification=True):

    features, labels, feature_names = read_file(dataset)

    if clf_name=='Torch':
        labels = labels.reshape(-1,1)
        features = features.astype(np.float32)
        if not classification:
            labels = labels.astype(np.float32)

    if classification:
        cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=random_state)
    else:
        cv = KFold(n_splits=10, shuffle=True,random_state=random_state)

    if classification:
        scoring = make_scorer(balanced_accuracy_score)
    else:
        scoring = 'r2'

    grid_clf = GridSearchCV(clf,cv=cv, param_grid=hyper_params,
            verbose=1,n_jobs=1,scoring=scoring,error_score=0.0)
    
    # print ( pipeline_components)
    # print(pipeline_parameters)
    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        # warnings.simplefilter('ignore')
        
        t0 = time.process_time()
        # generate cross-validated predictions for each data point using the best estimator 
        grid_clf.fit(features,labels)
        
        runtime = time.process_time() - t0

        best_est = grid_clf.best_estimator_
        
        # get the correlation structure of the data transformation 
        phi_cor = 0
        if type(best_est).__name__ == 'Feat':
            phi_cor = np.mean((1+np.corrcoef(best_est.transform(features)))/2) 
        elif clf_name == 'ElasticNet' or clf_name == 'SGD':
            phi_cor = np.mean((1+np.corrcoef(features))/2)

        # get the size of the final model
        model_size=0
        # pdb.set_trace()
        if 'Feat' in clf_name:
            # get_dim() here accounts for weights in umbrella ML model
            model_size = best_est.get_n_params()+best_est.get_dim()
        elif 'MLP' in clf_name:
            model_size = np.sum([c.size for c in best_est.coefs_]+
                          [c.size for c in best_est.intercepts_])
        elif 'Torch' in clf_name:
            model_size = best_est.module.get_n_params()
        elif hasattr(best_est,'coef_'):
            model_size = best_est.coef_.size
        else:
            model_size = features.shape[1]

        score = grid_clf.best_score_
        if 'Torch' in clf_name:     # just print grid search params, otherwise too overwhelming
            param_string = ','.join(['{}={}'.format(p, v) for p,v in grid_clf.best_params_.items()])
        else:
            param_string = ','.join(['{}={}'.format(p, v) for p,v in best_est.get_params().items()])

        # pdb.set_trace()
        out_text = '\t'.join([dataset.split('/')[-1].split('.')[0],
                              clf_name,
                              param_string,
                              str(random_state), 
                              # str(accuracy),
                              # str(macro_f1),
                              str(score),
                              str(phi_cor),
                              str(runtime),
                              str(model_size)])
        print(out_text)
        with open(save_file, 'a') as out:
            out.write(out_text+'\n')
        sys.stdout.flush()
        # evaluate_model(dataset, save_file, random_seed, clf)

        import pandas as pd

        df = pd.DataFrame(data=grid_clf.cv_results_)
        df['seed'] = random_state
        cv_save_name = save_file.split('.csv')[0]+'_cv_results.csv'
        import os.path
        if os.path.isfile(cv_save_name):
            # if exists, append
            df.to_csv(cv_save_name, mode='a', header=False, index=False)
        else:
            df.to_csv(cv_save_name, index=False)

