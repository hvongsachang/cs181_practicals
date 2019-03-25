import operator
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
import util
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Loading features
X_train = pickle.load( open( "X_train.p", "rb" ) )
global_feat_dict = pickle.load( open( "global_feat_dict.p", "rb" ) )
t_train = pickle.load( open( "t_train.p", "rb" ) )
train_ids = pickle.load( open( "train_ids.p", "rb" ) )

# Loading test data
X_test = pickle.load( open( "X_test.p", "rb" ) )
test_ids = pickle.load( open( "test_ids.p", "rb" ) )

# trying various values for penalty term C
parameters = {'C':[0.001, 0.01, 0.1, 1, 10]}

svc = SVC(decision_function_shape='ovo', verbose = 10)
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, t_train)

print clf.best_params_
print clf.best_score_
print clf.cv_results_

# trying again on more values for penalty term C
svc = SVC(decision_function_shape='ovo', verbose = 10)
clf = GridSearchCV(svc, param_grid={'C':[10, 15, 20, 50, 100]}, cv=5)
clf.fit(X_train, t_train)

print clf.best_params_
print clf.best_score_
print clf.cv_results_

# plotting validation curve
param_range = [0.001, 0.01, 0.1, 1, 10, 15, 20, 50, 100]
train_scores, test_scores = validation_curve(
    SVC(decision_function_shape='ovo'), X_train, t_train, param_name="C", param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# code for plotting curve credits to scikit-learn.org
plt.title("Validation Curve with SVM")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

clf_svm_gs = SVC(C = 20, decision_function_shape='ovo', verbose = 10)
clf_svm_gs.fit(X_train, t_train)
preds_svm_gs = clf_svm_gs.predict(X_test)
util.write_predictions(preds_svm_gs, test_ids, "SVM_GS_bigrams.csv")
files.download("SVM_GS_bigrams.csv")
