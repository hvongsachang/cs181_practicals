import operator
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
import util
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Loading features

X_train = pickle.load( open( "X_train.p", "rb" ) )
global_feat_dict = pickle.load( open( "global_feat_dict.p", "rb" ) )
t_train = pickle.load( open( "t_train.p", "rb" ) )
train_ids = pickle.load( open( "train_ids.p", "rb" ) )

# Loading test data
X_test = pickle.load( open( "X_test.p", "rb" ) )
test_ids = pickle.load( open( "test_ids.p", "rb" ) )

# Grid search CV to optimize alpha and learning rate
parameters = {'alpha':[0.01, 0.001, 0.0001, 0.00001], 'learning_rate_init':[0.01, 0.001, 0.0001, 0.00001]}
mlp = MLPClassifier(verbose = 10)
clf = GridSearchCV(mlp, parameters, cv=5) # 5-fold
clf.fit(X_train, t_train)

print clf.best_params_
print clf.best_score_
print clf.cv_results_

# refit again on best hyperparameters
clf_mlp = MLPClassifier(alpha = 0.01, learning_rate_init = 0.001, verbose=10)
clf_mlp.fit(X_train, t_train)
print clf_mlp.score(X_train, t_train)

preds_mlp = clf_mlp.predict(X_test)
util.write_predictions(preds_mlp, test_ids, "MLP_GS_bigrams.csv")
files.download("MLP_GS_bigrams.csv")