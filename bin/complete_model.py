#!/usr/bin/env python3

import os
import sys
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from itertools import product

sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'notebooks'))
from pgv_ml_utils import *


###############  Generate list of samples by event
pq_list = [os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/', f) \
           for f in os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/'))]
meta_list = [read_meta(pq_fs) for pq_fs in pq_list if ".pq" in pq_fs]
meta_df = pd.DataFrame.from_records(meta_list)
######################
# ordered event list to roughly distribute testing
ambient_list = list(meta_df[meta_df.magnitude.isnull()].eventID.unique())
event_list = meta_df[~meta_df.magnitude.isnull()].sort_values(['magnitude'], ascending=False).groupby(
    "eventID").count().sort_values(['station'], ascending=False).index.tolist()
full_list = ambient_list + event_list

######Hyperparameters to grid search over

d = {'n_folds': [5], 'max_depth': [10, 100], 'n_estimators': [3, 100, 200], 'class_wt': [None, "balanced_subsample"]}
hyperp = [dict(zip(d, v)) for v in product(*d.values())]

fs = {'dims': [['e', 'n', 'u']], 'stacking': ['horizontal']}
feature_sets = [dict(zip(fs, v)) for v in product(*fs.values())]

########

for features in feature_sets:
    params = [i | features for i in hyperp]
    best_est_, stats = grid_search(full_list, params)

    X_train, y_train, name_list, times = list_to_featurearrays(full_list, best_est_)
    clf = RandomForestClassifier(n_estimators=best_est_['n_estimators'], max_depth=best_est_['max_depth'],
                                 class_weight=best_est_['class_wt'], random_state=10, n_jobs=-1).fit(X_train, y_train)

    keep_thresh = str(int(100 * stats.threshold))

    joblib.dump(clf, os.path.join(os.path.dirname(os.getcwd()), 'data/model_all_%s.pkl' % keep_thresh))
