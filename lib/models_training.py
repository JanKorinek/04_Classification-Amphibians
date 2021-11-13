#!/usr/bin/python
"""
Collection of functions for training multiple models with RandomizedSearch .
"""
# Libraries import
import time, joblib, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import cycle
from scipy import interp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.utils.fixes import loguniform

# Warnings turn off
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')

pd.set_option("display.max.columns", None)
plt.style.use('seaborn')

SMALL_SIZE = 15
MEDIUM_SIZE = 18
LARGE_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

# Model specific variables
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "Supervised learning model for Amphibians classification"

def plot_roc(y_test, y_pred, preffix):
    """
    Funtion plots ROC curves for predicted classes and mutually compares them.
    :param y_test: test data array
    :param y_pred: predicted data array
    :param preffix: (str) model name
    :return: ROC curves plot, FPs, TPs and ROC score
    """
    # Compute ROC curve and ROC area for each class
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test.values[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.values.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(15, 15))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["blue", "orange", "green", 'red', 'purple', 'brown', 'cyan'])
    for i, class_name, color in zip(range(n_classes), y_test.columns, colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(class_name, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate -->", fontsize=LARGE_SIZE, )
    plt.ylabel("True Positive Rate -->", fontsize=LARGE_SIZE, )
    plt.title(f" {preffix} ROC of classes", fontsize=LARGE_SIZE, fontweight='bold')
    plt.legend(loc="lower right", fontsize=MEDIUM_SIZE, )
    plt.tight_layout()
    plt.savefig(f'export/roc_classes_{preffix}_comparison.pdf', dpi=600)

    plt.show()

    return fpr, tpr, roc_auc


def compare_roc(scoring):
    """
    Function plots ROC curves of all trained models
    :param scoring: (list) Scores of all models
    :return: Plot
    """
    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(15, 15))

    colors = cycle(["blue", "orange", "green", 'red'])
    names = ['Logistic', 'KNeighbor', 'Decision Tree', 'Random Forest']

    for i, name, color in zip(scoring, names, colors):
        fpr = i[0]['micro']
        tpr = i[1]['micro']
        roc_auc = i[2]['micro']
        plt.plot(fpr, tpr, color=color, lw=lw,
            label="{0} model ROC curve (area = {1:0.2f})".format(name, roc_auc),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate -->", fontsize=LARGE_SIZE, )
    plt.ylabel("True Positive Rate -->", fontsize=LARGE_SIZE, )
    plt.title("ROC Curves Comparison ", fontsize=LARGE_SIZE, fontweight='bold')
    plt.legend(loc="lower right", fontsize=MEDIUM_SIZE, )
    plt.tight_layout()
    plt.savefig(f'export/roc_models_comparison.pdf', dpi=600)

    plt.show()

def train_model(X_train, X_test, y_train, y_test, pipe, param_grid,
                             params_gen, preffix):
    """
    Funtion to train the model
    """
    print(f'\nTraining the {preffix} model...\n')

    # Start timer for runtime
    time_start = time.time()

    # Defining splitter
    splitter = KFold(n_splits=params_gen['n_splits'])

    # Define scoring metrics
    scoring = ['f1_samples', 'accuracy', 'roc_auc']

    rand = RandomizedSearchCV(pipe,
                              param_distributions = param_grid,
                              n_iter = params_gen['n_iter'],
                              cv = splitter,
                              verbose=0,
                              random_state=42,
                              n_jobs = -1,
                              scoring=scoring,
                              refit='f1_samples',)
    rand.fit(X_train, y_train)
    y_pred = rand.predict(X_test)

    # Evaluation metrics
    score_acc = accuracy_score(y_test,y_pred)
    score_f1 = f1_score(y_test, y_pred, average='micro')
    score_roc = roc_auc_score(y_test, y_pred, average='macro')
    eval = [score_acc, score_f1, score_roc]

    # Plot ROC curves
    fps, tps, roc = plot_roc(y_test, y_pred ,preffix)
    scoring = [fps, tps, roc]

    # Plotting additional information
    print(f'\n------ The best parameters of the {preffix} model are: ------')
    print(rand.best_estimator_)
    print('-'*50)
    print(f'The best cross-validation score: {rand.best_score_}')
    print(f'Accuracy: {score_acc}')
    print(f'F1: {score_f1}')
    print(f'ROC_AUC: {score_roc}')
    print('-' * 50)

    # Save the model
    joblib.dump(rand, f'models/{preffix}_model.joblib')

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    print(f'\n{preffix} model training finished in:', '%d:%02d:%02d'%(h, m, s))

    return rand, eval, y_pred, scoring


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Training all models...\n')

    # General Parameters
    input_params = {
        'n_splits': 5,
        'n_iter': 32,
    }

    # Data import
    amph = pd.read_pickle('data/amphibians_oversample.pickle')

    # Prepare train-test split
    target_columns = ['Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad',
                      'Tree frog', 'Common newt', 'Great crested newt']

    X = amph.drop(columns=target_columns)
    y = amph[target_columns]  # Targets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, shuffle=True)

    # Logistic classifier RandomSearch, hyperparameters and pipeline
    clf = MultiOutputClassifier(LogisticRegression())

    param_rand_l = {
        'clf__estimator__C': loguniform(1e-4, 1e4),
        'clf__estimator__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'clf__estimator__class_weight': ['balanced', None],
    }
    pipe_l = Pipeline(steps=[('scaler', StandardScaler()),
                            ('clf', MultiOutputClassifier(LogisticRegression()))])
    # Train the model
    l_model, l_eval, l_y_pred, l_scoring = train_model(X_train, X_test, y_train,
                                                       y_test, pipe_l, param_rand_l,
                                                       input_params, 'Logistic')

    # KNeighbor classifier RandomSearch, hyperparameters and pipeline
    clf = MultiOutputClassifier(KNeighborsClassifier())

    param_rand_kn = {
        'clf__estimator__n_neighbors': list(range(1,30)),
        'clf__estimator__leaf_size': list(range(1,50)),
        'clf__estimator__p': [1,2],
        'clf__estimator__weights': ['uniform', 'distance'],
        'clf__estimator__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'clf__estimator__metric': ['euclidean', 'minkowski'],
    }
    pipe_kn = Pipeline(steps=[('scaler', StandardScaler()),
                            ('clf', MultiOutputClassifier(KNeighborsClassifier()))])
    # Train the model
    kn_model, kn_eval, kn_y_pred, kn_scoring = train_model(X_train, X_test, y_train,
                                                           y_test, pipe_kn, param_rand_kn,
                                                           input_params, 'KNeighbor')

    # Decision Tree classifier RandomSearch, hyperparameters and pipeline
    clf = MultiOutputClassifier(DecisionTreeClassifier())

    param_rand_dt = {
            'clf__estimator__criterion': ['gini', 'entropy'],
            'clf__estimator__max_features': ['auto', 'sqrt'],
            'clf__estimator__min_samples_split': range(1,10),
            'clf__estimator__min_samples_leaf': range(1,10),
            'clf__estimator__class_weight': ['balanced', None],
    }
    pipe_dt = Pipeline(steps=[('scaler', StandardScaler()),
                            ('clf', MultiOutputClassifier(DecisionTreeClassifier()))])
    # Train the model
    dt_model, dt_eval, dt_y_pred, dt_scoring = train_model(X_train, X_test, y_train,
                                                           y_test, pipe_dt, param_rand_dt,
                                                           input_params, 'Decision_Tree')

    # RandomForest classifier RandomSearch, hyperparameters and pipeline
    clf = MultiOutputClassifier(RandomForestClassifier())
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    param_rand_rf = {
        'clf__estimator__bootstrap': [True, False],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__n_estimators': [int(x) for x in np.linspace(start=50, stop=1000, num=20)],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__max_depth': max_depth,
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__class_weight': ['balanced', 'balanced_subsample', None],
    }
    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                            ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # Train the model
    rf_model, rf_eval, rf_y_pred, rf_scoring = train_model(X_train, X_test, y_train,
                                                           y_test, pipe_rf, param_rand_rf,
                                                           input_params, 'Random_Forest')

    # Plot model ROC curves comparison
    scoring = [l_scoring, kn_scoring, dt_scoring, rf_scoring]
    compare_roc(scoring)

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('\nAll models trained in:', '%d:%02d:%02d'%(h, m, s))
