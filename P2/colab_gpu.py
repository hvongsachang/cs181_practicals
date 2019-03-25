from classification_starter_remove2 import extract_feats, first_last_system_call_feats, system_call_count_feats
from feature_selection import merge_functions, find_top_100, get_word_counts, topic_models, get_process_times_and_sizes
from sklearn.ensemble import RandomForestClassifier
import operator
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm
import util
import pandas as pd
import numpy as np

def feature_importance(X_train, t_train, features):
	print "training..."
	clf = RandomForestClassifier(n_estimators = 1024, max_depth = 24, random_state=37)
	clf.fit(X_train, t_train)

	print "feature importance..."
	print sorted(zip(features, clf.feature_importances_), key=operator.itemgetter(1), reverse=True)

def depth_tuning(X_train, t_train, features, depth):
	r2_train_rf = []
	rocs = []
	print "depth tuning..."
	for i in tqdm(range(1, depth)):
	    rf_reg = RandomForestClassifier(max_depth=i)
	    rf_reg.fit(X_train, t_train)
	    rf_yhat_train = rf_reg.predict(X_train)

	    r2_train_rf.append(r2_score(t_train, rf_yhat_train))
	    # rocs.append(roc_auc_score(t_train, rf_yhat_train))

	print "R^2 Values for max_depth"
	print sorted(zip(range(1,depth), r2_train_rf), key=operator.itemgetter(1), reverse=True)

	# print "ROC"
	# print sorted(zip(range(1,depth), rocs), key=operator.itemgetter(1), reverse=True)
	return

def estimator_tuning(X_train, t_train, features):
	r2_train_rf = []
	print "estimator tuning..."
	est = [32,64,128,256,512,1024]
	for i in tqdm(est):
	    rf_reg = RandomForestClassifier(n_estimators=i, max_depth = 26)
	    rf_reg.fit(X_train, t_train)
	    rf_yhat_train = rf_reg.predict(X_train)

	    r2_train_rf.append(r2_score(t_train, rf_yhat_train))

	print "R^2 Values for n_estimators"
	print sorted(zip(range(1,est), r2_train_rf), key=operator.itemgetter(1), reverse=True)
	return

def random_search_tuning(X_train, t_train):
	clf = RandomForestClassifier(random_state=97)

	param_dist = { 
		"n_estimators": [32,64,128,256,512,1024], 
		"max_depth": [23,26,21,30],
	}
	# param_dist = { 
	# 	"n_estimators": [32,64,128,256,512,1024], 
	# 	"max_depth": [23,24,21,27,30],
	# }
	#param_dist = { "n_estimators": [64,96,128,192,256], "max_depth": [50,53,33,57,34,39]}

	print "random search"
	random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=30, cv=10, verbose=10)
	random_search.fit(X_train, t_train)
	# print pd.DataFrame(random_search.cv_results_)
	print random_search.cv_results_
	return

def load_train():
	X_train = pickle.load( open( "X_train.p", "rb" ) )
	global_feat_dict = pickle.load( open( "global_feat_dict.p", "rb" ) )
	t_train = pickle.load( open( "t_train.p", "rb" ) )
	train_ids = pickle.load( open( "train_ids.p", "rb" ) )

	sorted_x = sorted(global_feat_dict.items(), key=operator.itemgetter(1))
	features = [x[0] for x in sorted_x]

	return X_train, global_feat_dict, t_train, train_ids, features

def update_feature_classes(ffs, name):
	train_dir = "train"
	test_dir = "test"

	X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)

	pickle.dump( X_train, open( "X_train_" + str(name) + ".p", "wb" ) )
	pickle.dump( global_feat_dict, open( "global_feat_dict" + str(name) + ".p", "wb" ) )
	pickle.dump( t_train, open( "t_train_enhanced_"  + str(name) + ".p", "wb" ) )
	pickle.dump( train_ids, open( "train_ids_enhanced_" + str(name) + ".p", "wb" ) )

	del X_train
	del t_train
	del train_ids

	X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)

	pickle.dump( X_test, open( "X_test_" + str(name) + ".p", "wb" ) )
	pickle.dump( t_ignore, open( "t_ignore_" + str(name) + ".p", "wb" ) )
	pickle.dump( test_ids, open( "test_ids_" + str(name) + ".p", "wb" ) )
	print "done"
	return

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
		X_train = pickle.load( open( "/content/drive/P2/X_train_enhanced.p", "rb" ) )
		global_feat_dict = pickle.load( open( "/content/drive/P2/global_feat_dictenhanced.p", "rb" ) )
		t_train = pickle.load( open( "/content/drive/P2/t_train_enhanced_enhanced.p", "rb" ) )
		train_ids = pickle.load( open( "/content/drive/P2/train_ids_enhanced_enhanced.p", "rb" ) )
	print "done extracting training features"
	print

	# print "sorting features"
	sorted_x = sorted(global_feat_dict.items(), key=operator.itemgetter(1))
	features = [x[0] for x in sorted_x]

	# # print "features"
	# # print features

	# feature_importance(X_train, t_train, features)
	# depth_tuning(X_train, t_train, features, 32)
	# estimator_tuning(X_train, t_train, features)
	random_search_tuning(X_train, t_train)

	### INSERT CLASSIFICATION HERE
	# print "classifying..."
	# clf = RandomForestClassifier(n_estimators = 1024, max_depth = 24, random_state=37)
	# clf.fit(X_train, t_train)
	### INSERT CLASSIFICATION HERE

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
		X_test = pickle.load( open( "/content/drive/P2/X_test_enhanced.p", "rb" ) )
		t_ignore = pickle.load( open( "/content/drive/P2/t_ignore_enhanced.p", "rb" ) )
		test_ids = pickle.load( open( "/content/drive/P2/test_ids_enhanced.p", "rb" ) )
	print "done extracting test features"
	print

	# print "making predictions..."
	# preds = clf.predict(X_test)
	# print "done making predictions"
	# print

	# print "writing predictions..."
	# util.write_predictions(preds, test_ids, "RF_grams.csv")
	# print "done!"

	# # preds = clf.predict(X_train)
	# # print confusion_matrix(t_train, preds)
	return

def checking_feature_selection_function(funcs, should_reload = False):
	train_dir = "train"
	test_dir = "test"

	if should_reload:
		ffs = funcs
		X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
		return

	print "topic models"
	print topic_models()
	return
		

def remove_high_correlation(X_train):
	X_train_pd = pd.DataFrame(X_train)
	corr_matrix = X_train_pd.corr().abs()
	high_corr_var=np.where(corr_matrix>0.999)
	high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]

	remove_vars = [i[0] for i in high_corr_var]
	print(remove_vars)
	return
    
    # X_corr_train = X_train.drop(columns=remove_vars)
    # X_corr_test = X_test.drop(columns=remove_vars)
    # return X_corr_train, X_corr_test

def main():
	ffs = [first_last_system_call_feats, system_call_count_feats, merge_functions, get_word_counts, topic_models]

	# testing individual functions in feature_selection.py
	### UNCOMMENT BELOW FOR TESTING
	# print checking_feature_selection_function(ffs)#, should_reload = True)
	### UNCOMMENT ABOVE FOR TESTING

	### UNCOMMENT BELOW FOR TESTING
	# print "loading..."
	# X_train, global_feat_dict, t_train, train_ids, features = load_train()
	# print pd.DataFrame(X_train)
	# print "feature importance..."
	# feature_importance(X_train, t_train, features)
	# remove_high_correlation(X_train)
	### UNCOMMENT ABOVE FOR TESTING

	# ready for final prediction
	### UNCOMMENT BELOW FOR PREDICTING
	print predict_classes(ffs)
	# update_feature_classes(ffs, "enhanced")
	### UNCOMMENT ABOVE FOR PREDICTING
	
	return

if __name__ == "__main__":
    main()