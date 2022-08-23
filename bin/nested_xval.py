#!/usr/bin/env python3

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import precision_recall_curve

from itertools import product
from collections import deque

sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd()), 'notebooks'))
from pgv_ml_utils import *

## this script runs nested cross validation (10 outer folds)
## each test fold runs a grid search over a range of hyperparameters and feature vectors
## each grid search is performed using an inner loop of k-fold cross validation
## the horizontal stacking model (previously determined to be optimal) is stored for each run for future testing
## the results of the 10 folds are stored in a .csv
## the 10 fold testing concatenated target and feature vectors are stored in a .pkl file

def nested_xval():

    #pq_list is all available parquet files (event/station) of featuresets
    pq_list = [os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/', f) for f in
               os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/'))]
    # meta_list compiles all available metadata attached to parquet featureset files
    meta_list = [read_meta(pq_fs) for pq_fs in pq_list if ".pq" in pq_fs]
    meta_df = pd.DataFrame.from_records(meta_list)

    ######################
    # ordered event list to roughly distribute events in testing
    ambient_list = list(meta_df[meta_df.magnitude.isnull()].eventID.unique())
    event_list = meta_df[~meta_df.magnitude.isnull()].sort_values(['magnitude'], ascending=False).groupby(
        "eventID").count().sort_values(['station'], ascending=False).index.tolist()
    full_list = ambient_list + event_list

    #######################

    d = {'n_folds': [5], 'max_depth': [10, 100], 'n_estimators': [3, 100, 200], 'class_wt': [None, "balanced_subsample"]}
    hyperp = [dict(zip(d, v)) for v in product(*d.values())]

    fs = {'dims': [['e', 'n', 'u']], 'stacking': ['horizontal', 'vertical']}
    feature_sets = [dict(zip(fs, v)) for v in product(*fs.values())]

    y_pred_keep = []
    y_test_keep = []
    x_test_keep = []

    outer_results = []
    # Nested Cross validation, 10 runs
    num_runs = 10
    for k in np.arange(num_runs):
        run = k + 1
        items = deque(full_list)
        items.rotate(-k)
        test_set = list(items)[::num_runs]
        train_set = list(set(full_list) - set(test_set))

        for features in feature_sets:

            params = [i | features for i in hyperp]
            best_est_, stats = grid_search(train_set, params)

            X_train, y_train, name_list, times = list_to_featurearrays(train_set, best_est_)
            X_test, y_test, name_list, times = list_to_featurearrays(test_set, best_est_)
            clf = RandomForestClassifier(n_estimators=best_est_['n_estimators'], max_depth=best_est_['max_depth'],
                                         class_weight=best_est_['class_wt'], random_state=10, n_jobs=-1).fit(X_train,
                                                                                                             y_train)
            y_pred_prob = clf.predict_proba(X_test)[:, 1]
            threshold = stats.threshold  # Hyper Param from xval training
            y_pred = (y_pred_prob >= threshold).astype('int')

            ###
            # evaluate the model on test data
            p, r, f1, blah = precision_recall_fscore_support(y_test, y_pred, average='binary')
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

            # store the result
            outer_results.append([p, r, f1, threshold, precisions, recalls, thresholds, y_test, y_pred_prob, best_est_, run,
                                  features['stacking'], test_set])
            # report progress
            print('>f1=%.3f, %s' % (f1, stats))
            executionTime = (time.time() - startTime)
            print('Execution time in seconds: ' + str(executionTime))
            if features['stacking'] == 'horizontal':
                joblib.dump(clf, os.path.join(os.path.dirname(os.getcwd()), 'data/model_run_%s.pkl' % run))

                y_pred_keep.append(y_pred)
                y_test_keep.append(y_test)
                x_test_keep.append(X_test)
    df = pd.DataFrame(outer_results,
                      columns=['precision', 'recall', 'f1', 'threshold', 'precisions', 'recalls', 'thresholds', 'y_act',
                               'y_prob', 'params', 'run', 'stacking', 'test stations'])

    df.to_csv('nested_x_val1_final.csv')

    y_pred_data = (np.concatenate(y_pred_keep, axis=0))
    y_test_data = (np.concatenate(y_test_keep, axis=0))
    x_test_data = np.concatenate(x_test_keep, axis=0)

    ydf = pd.DataFrame([y_pred_data, y_test_data]).T
    xdf = pd.DataFrame(x_test_data)

    ydf.to_pickle('ydf.pkl')
    xdf.to_pickle('xdf.pkl')

    plot_results(df)

if __name__ == "__main__":
    nested_xval()



def plot_results(df)
    # Function to plot/save feature engineering comparison results
    # takes pandas dataframe of nested xval results
    # generates/saves 2x1 plot of Precision, Recall and F1 scores


    ######################
    # set plotting params #
    #######################
    fsize = 15
    tsize = 18
    tdir = 'in'
    major = 5.0
    minor = 3.0
    lwidth = 0.8
    lhandle = 2.0
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = fsize - 3
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = 5.0
    plt.rcParams['ytick.minor.size'] = 3.0
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['legend.handlelength'] = lhandle
    plt.rcParams.update(plt.rcParamsDefault)
    ###############

    bins = np.arange(-.1, 1, .1)
    # initialize
    fig, ax = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

    test_df = pd.DataFrame(data={'precisions': pd.Series(df[df.stacking == 'horizontal'].loc[0].precisions[:-1]),
                                 'recalls': pd.Series(df[df.stacking == 'horizontal'].loc[0].recalls[:-1]),
                                 'thresholds': pd.Series(df[df.stacking == 'horizontal'].loc[0].thresholds[:-1])})
    test_df['f1'] = (2 * test_df.precisions * test_df.recalls) / (test_df.precisions + test_df.recalls)

    test_df = test_df.set_index('thresholds')

    means = test_df.groupby(pd.cut(test_df.index, bins=bins)).mean()
    stdevs = test_df.groupby(pd.cut(test_df.index, bins=bins)).std()

    print(means)
    # fill
    for ix in np.arange(2, 20, 2):
        test_df = pd.DataFrame(data={'precisions': df[df.stacking == 'horizontal'].loc[ix].precisions[:-1],
                                     'recalls': df[df.stacking == 'horizontal'].loc[ix].recalls[:-1],
                                     'thresholds': df[df.stacking == 'horizontal'].loc[ix].thresholds})
        test_df = test_df.set_index('thresholds')

        b = test_df.groupby(pd.cut(test_df.index, bins=bins)).mean()
        c = test_df.groupby(pd.cut(test_df.index, bins=bins)).std()
        means += b
        stdevs += c

    means = means / num_runs
    print(means)
    stdevs = stdevs / num_runs

    color = ['#e41a1c', '#377eb8', '#4daf4a']
    ax[0].plot(bins[1:], means.precisions, label='Precision', color=color[0])
    ax[0].fill_between(bins[1:], means.precisions - stdevs.precisions, means.precisions + stdevs.precisions, alpha=.3,
                       color=color[0])
    ax[0].plot(bins[1:], means.recalls, label='Recall', color=color[1])
    ax[0].fill_between(bins[1:], means.recalls - stdevs.recalls, means.recalls + stdevs.recalls, alpha=.3, color=color[1])

    ax[0].plot(bins[1:], (2 * means.precisions * means.recalls) / (means.precisions + means.recalls), label='F1',
               color=color[2], lw=2)
    ax[0].axvline(bins[((2 * means.precisions * means.recalls) / (means.precisions + means.recalls)).argmax() + 1],
                  linestyle='dashed', color='gray')
    ax[0].axhline(((2 * means.precisions * means.recalls) / (means.precisions + means.recalls)).max(),
                  linestyle='dashed', color='red')
    ax[0].legend(loc='lower right')

    ax[0].set_xlim([bins[4], bins[-1]])
    ax[0].set_title('Feature Set 1')
    ax[0].set_xlabel('Decision threshold')
    ax[0].set_ylabel('Score')

    ##TEST 2#############

    test_df = pd.DataFrame(data={'precisions': df[df.stacking == 'vertical'].loc[1].precisions[:-1],
                                 'recalls': df[df.stacking == 'vertical'].loc[1].recalls[:-1],
                                 'thresholds': df[df.stacking == 'vertical'].loc[1].thresholds})
    test_df = test_df.set_index('thresholds')

    means = test_df.groupby(pd.cut(test_df.index, bins=bins)).mean()
    stdevs = test_df.groupby(pd.cut(test_df.index, bins=bins)).std()
    # fill
    for ix in np.arange(3, 20, 2):
        test_df = pd.DataFrame(data={'precisions': df[df.stacking == 'vertical'].loc[ix].precisions[:-1],
                                     'recalls': df[df.stacking == 'vertical'].loc[ix].recalls[:-1],
                                     'thresholds': df[df.stacking == 'vertical'].loc[ix].thresholds})
        test_df = test_df.set_index('thresholds')

        b = test_df.groupby(pd.cut(test_df.index, bins=bins)).mean()
        c = test_df.groupby(pd.cut(test_df.index, bins=bins)).std()
        means += b
        stdevs += c

    means = means / num_runs
    stdevs = stdevs / num_runs

    ax[1].plot(bins[1:], means.precisions, label='Precision', color=color[0])
    ax[1].fill_between(bins[1:], means.precisions - stdevs.precisions, means.precisions + stdevs.precisions, alpha=.3,
                       color=color[0])
    ax[1].plot(bins[1:], means.recalls, label='Recall', color=color[1])
    ax[1].fill_between(bins[1:], means.recalls - stdevs.recalls, means.recalls + stdevs.recalls, alpha=.3, color=color[1])

    ax[1].plot(bins[1:], (2 * means.precisions * means.recalls) / (means.precisions + means.recalls), label='F1',
               color=color[2], lw=2)

    ax[1].axvline(bins[((2 * means.precisions * means.recalls) / (means.precisions + means.recalls)).argmax() + 1],
                  linestyle='dashed', color='gray')
    ax[1].axhline(((2 * means.precisions * means.recalls) / (means.precisions + means.recalls)).max(),
                  linestyle='dashed', color='red')
    ax[1].legend(loc='lower right')

    ax[1].set_title('Feature Set 2')
    ax[1].set_xlabel('Decision threshold')
    ax[1].set_xlim([0.1, .8])
    ax[0].set_xlim([0.1, .8])

    ax[1].set_ylim([0.3, 1])

    plt.subplots_adjust(wspace=0.1)

    plt.savefig('fs_evaluation.png', dpi=500)


