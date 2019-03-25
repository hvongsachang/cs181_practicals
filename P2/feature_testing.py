from classification_starter_remove2 import extract_feats, first_last_system_call_feats, system_call_count_feats
from feature_selection import merge_functions, find_top_100, get_word_counts, topic_models, get_header_tags
from sklearn.ensemble import RandomForestClassifier
import operator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
import util
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import math
import itertools
from sklearn.linear_model import LogisticRegression

def feature_importance(X_train, t_train, features):
	print "training..."
	clf = RandomForestClassifier(n_estimators = 128, max_depth = 24, random_state=97)
	clf.fit(X_train, t_train)

	print "feature importance..."
	print sorted(zip(features, clf.feature_importances_), key=operator.itemgetter(1), reverse=True)

def depth_tuning(X_train, t_train, depth):
	r2_train_rf = []
	rocs = []
	print "depth tuning..."
	for i in tqdm(range(1, depth)):
	    rf_reg = RandomForestClassifier(max_depth=i)
	    rf_reg.fit(X_train, t_train)
	    rf_yhat_train = rf_reg.predict(X_train)

	    r2_train_rf.append(r2_score(t_train, rf_yhat_train))

	print "R^2 Values for max_depth"
	print sorted(zip(range(1,depth), r2_train_rf), key=operator.itemgetter(1), reverse=True)
	return

def estimator_tuning(X_train, t_train):
	r2_train_rf = []
	print "estimator tuning..."
	est = [32,64,128,256,512,1024]
	for i in tqdm(est):
	    rf_reg = RandomForestClassifier(n_estimators=i, max_depth = 23)
	    rf_reg.fit(X_train, t_train)
	    rf_yhat_train = rf_reg.predict(X_train)

	    r2_train_rf.append(r2_score(t_train, rf_yhat_train))

	print "R^2 Values for n_estimators"
	print sorted(zip(est, r2_train_rf), key=operator.itemgetter(1), reverse=True)
	return

def random_search_tuning(X_train, t_train):
	clf = RandomForestClassifier(random_state=97)

	param_dist = { 
		"n_estimators": [128,256,512,1024], 
		"max_depth": [20,23,26],
	}

	print "random search..."
	random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=30, cv=10, verbose=10)
	random_search.fit(X_train, t_train)
	print random_search.cv_results_
	return

def load_train():
	X_train = pickle.load( open( "X_train_enhanced2.p", "rb" ) )
	global_feat_dict = pickle.load( open( "global_feat_dict_enhanced2.p", "rb" ) )
	t_train = pickle.load( open( "t_train_enhanced2.p", "rb" ) )
	train_ids = pickle.load( open( "train_ids_enhanced2.p", "rb" ) )

	sorted_x = sorted(global_feat_dict.items(), key=operator.itemgetter(1))
	features = [x[0] for x in sorted_x]

	return X_train, global_feat_dict, t_train, train_ids, features

def update_feature_classes(ffs, name):
	train_dir = "train"
	test_dir = "test"

	X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)

	pickle.dump( X_train, open( "X_train_" + str(name) + ".p", "wb" ) )
	pickle.dump( global_feat_dict, open( "global_feat_dict_" + str(name) + ".p", "wb" ) )
	pickle.dump( t_train, open( "t_train_"  + str(name) + ".p", "wb" ) )
	pickle.dump( train_ids, open( "train_ids_" + str(name) + ".p", "wb" ) )

	del X_train
	del t_train
	del train_ids

	X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)

	pickle.dump( X_test, open( "X_test_" + str(name) + ".p", "wb" ) )
	pickle.dump( t_ignore, open( "t_ignore_" + str(name) + ".p", "wb" ) )
	pickle.dump( test_ids, open( "test_ids_" + str(name) + ".p", "wb" ) )
	print "done"
	return

def group_by_features(X_train_pd, global_feat_dict, pairs):
	for pair in pairs:
		for i in range(2):
			if i == 0:
				a, b = pair
			else:
				b, a = pair
			feature = X_train_pd.groupby(a)[b].mean()
			vals = feature.index.tolist()
			n_mean = list(feature)
			assoc_dict = dict(zip(vals, n_mean))

			agg_features = []
			for num in X_train_pd[a]:
				val = assoc_dict[num]
				agg_features.append(val)
			X_train_pd[(a,b)] = agg_features

	return X_train_pd


def predict_classes(ffs):

	train_dir = "train"
	test_dir = "test"

	# change when updating features
	trained_already = True

	print "extracting training features..."
	if not trained_already:
		X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)

		pickle.dump( X_train, open( "X_train.p", "wb" ) )
		pickle.dump( global_feat_dict, open( "global_feat_dict.p", "wb" ) )
		pickle.dump( t_train, open( "t_train.p", "wb" ) )
		pickle.dump( train_ids, open( "train_ids.p", "wb" ) )

	else:
		X_train = pickle.load( open( "X_train_enhanced2.p", "rb" ) )
		global_feat_dict = pickle.load( open( "global_feat_dict_enhanced2.p", "rb" ) )
		t_train = pickle.load( open( "t_train_enhanced2.p", "rb" ) )
		train_ids = pickle.load( open( "train_ids_enhanced2.p", "rb" ) )
	print "done extracting training features"
	print

	# GET FEATURES BELOW
	sorted_x = sorted(global_feat_dict.items(), key=operator.itemgetter(1))
	features = [x[0] for x in sorted_x]
	# GET FEATURES ABOVE

	print "NUMBER OF FEATURES: " + str(len(features))

	### TESTING GROUP BY BELOW
	group_features = ['create_thread_remote', 'vm_allocate', 'show_window', 'vm_write', 'read_value']
	feature_indices = []

	print "finding feature values"
	for g in group_features:
		if g in global_feat_dict:
			feature_indices.append(global_feat_dict[g])

	X_train_pd = pd.DataFrame(X_train.toarray())

	pairs = []
	for pair in itertools.combinations(feature_indices,2):
		pairs.append(pair)

	X_train = group_by_features(X_train_pd, global_feat_dict, pairs)
	### TESTING GROUP BY ABOVE

	### INSERT CLASSIFICATION BELOW HERE
	print "classifying..."
	clf = RandomForestClassifier(n_estimators = 1024, max_depth = 23, random_state=97)
	clf.fit(X_train_pd, t_train)
	### INSERT CLASSIFICATION ABOVE HERE

	del X_train
	del t_train
	del train_ids

	print "extracting test features..."
	if not trained_already:
		X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)

		pickle.dump( X_test, open( "X_test.p", "wb" ) )
		pickle.dump( t_ignore, open( "t_ignore.p", "wb" ) )
		pickle.dump( test_ids, open( "test_ids.p", "wb" ) )
	else:
		X_test = pickle.load( open( "X_test_enhanced2.p", "rb" ) )
		t_ignore = pickle.load( open( "t_ignore_enhanced2.p", "rb" ) )
		test_ids = pickle.load( open( "test_ids_enhanced2.p", "rb" ) )
	print "done extracting test features"
	print

	### UNCOMMENT GROUP BY BELOW
	# X_test_pd = pd.DataFrame(X_test.toarray())
	# X_test = group_by_features(X_test_pd, global_feat_dict, pairs)
	### UNCOMMENT GROUP BY ABOVE

	### UNCOMMENT FOR INDIVIDUAL RF BELOW
	# print "classifying individual..."
	# predictions_probs = []
	# for i in tqdm(range(15)):
	# 	single = [1 if x == i else 0 for x in t_train]
	# 	clf = RandomForestClassifier(n_estimators = 1024, max_depth = 23, random_state=97)
	# 	clf.fit(X_train, single)
	# 	pred_prob = clf.predict_proba(X_test)
	# 	this_class_probs = np.array(pred_prob)[:,1]
	# 	predictions_probs.append(this_class_probs)

	# final_pred_probs = pd.DataFrame(predictions_probs)
	# cols = final_pred_probs.columns
	# final_predictions = []
	# for c in cols:
	# 	final_predictions.append(final_pred_probs[c].idxmax())
	# print list(final_predictions)
	### UNCOMMENT FOR INDIVIDUAL RF ABOVE

	### UNCOMMENT BELOW FOR CONFUSION MATRIX
	# matrix = confusion_matrix(t_train, final_predictions)
	# pickle.dump( matrix, open( "confusion_matrix_logistic.p", "wb" ) )
	### UNCOMMENT ABOVE FOR CONFUSION MATRIX

	print "making predictions..."
	preds = clf.predict(X_test)
	print preds
	print "done making predictions"

	print "writing predictions..."
	util.write_predictions(final_predictions, test_ids, "individual_RF_1024.csv")
	print "done!"

	return

def scale_and_pca(X_train, X_test):
	scaler = StandardScaler(with_mean=False)
	scaler.fit(X_train)
	X_train_scale = scaler.transform(X_train)
	X_test_scale = scaler.transform(X_test)

	pca = PCA(0.9999)
	pca.fit(X_train_scale.toarray())

	X_train_scale_pca = pca.transform(X_train_scale.toarray())
	X_test_scale_pca = pca.transform(X_test_scale.toarray())

	return X_train_scale_pca, X_test_scale_pca

def checking_feature_selection_function(funcs, should_reload = False):
	train_dir = "train"
	test_dir = "test"

	if should_reload:
		ffs = funcs
		X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
		return

	print get_header_tags()
	return

def remove_high_correlation(X_train):
	X_train_pd = pd.DataFrame(X_train)
	corr_matrix = X_train_pd.corr().abs()
	high_corr_var=np.where(corr_matrix>0.999)
	high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

	remove_vars = [i[0] for i in high_corr_var]
    
    X_corr_train = X_train.drop(columns=remove_vars)
    X_corr_test = X_test.drop(columns=remove_vars)
    return X_corr_train, X_corr_test

def main():
	ffs = [first_last_system_call_feats, system_call_count_feats, merge_functions, get_word_counts, topic_models]

	### UPDATE PICKLES BELOW
	# update_feature_classes(ffs, "enhanced")
	### UPDATE PICKLES ABOVE

	### UNCOMMENT BELOW FOR TESTING
	# print checking_feature_selection_function(ffs, should_reload = True)
	# get_header_tags()
	# find_top_100()
	### UNCOMMENT ABOVE FOR TESTING

	### UNCOMMENT BELOW FOR TESTING
	# print "loading..."
	# X_train, global_feat_dict, t_train, train_ids, features = load_train()
	# print pd.DataFrame(X_train)
	# print "feature importance..."
	# feature_importance(X_train, t_train, features)
	# remove_high_correlation(X_train)
	### UNCOMMENT ABOVE FOR TESTING

	### UNCOMMENT BELOW FOR PREDICTING
	print predict_classes(ffs)
	### UNCOMMENT ABOVE FOR PREDICTING
	
	return

if __name__ == "__main__":
    main()