import operator
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
import util
from sklearn.linear_model import LogisticRegression

# Loading features

X_train = pickle.load( open( "X_train.p", "rb" ) )
global_feat_dict = pickle.load( open( "global_feat_dict.p", "rb" ) )
t_train = pickle.load( open( "t_train.p", "rb" ) )
train_ids = pickle.load( open( "train_ids.p", "rb" ) )

# Loading test data
X_test = pickle.load( open( "X_test.p", "rb" ) )
test_ids = pickle.load( open( "test_ids.p", "rb" ) )

# LBFGS
clf_lbfgs = LogisticRegression(solver="lbfgs", multi_class = "multinomial", max_iter=10000, verbose = 10)
clf_lbfgs.fit(X_train, t_train)
print clf_lbfgs.score(X_train, t_train)

# SAG
clf_sag = LogisticRegression(solver="sag", multi_class="multinomial", max_iter=10000, verbose = 10)
clf_sag.fit(X_train, t_train)
print clf_sag.score(X_train, t_train)

preds = clf_lbfgs.predict(X_test)
util.write_predictions(preds, test_ids, "LogReg_bigrams.csv")
files.download("LogReg_bigrams.csv")