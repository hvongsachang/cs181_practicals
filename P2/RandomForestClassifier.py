import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesClassifier

train_data_file = 'X_train.p'
train_classes_file = 't_train.p' 
test_data_file = 'X_test.p'
test_ids_file = 'test_ids.p'

def make_predictions():
  outputfile = "sample_predictions.csv"  # feel free to change this or take it as an argument
  
  # get rid of training data and load test data
  del X_train
  del t_train
  print "extracting test features..."
  X_test = pickle.load(open(test_data_file, 'rb'))
  test_ids = pickle.load(open(test_ids_file, 'rb'))
  print "done extracting test features"
  print
  
  # make predictions on text data and write them out
  print "making predictions..."
  preds = clf.predict(X_test)
  print "done making predictions"
  print
  
  print "writing predictions..."
  util.write_predictions(preds, test_ids, outputfile)
  print "done!"

## The following function does the feature extraction, learning, and prediction
def main():
    
    # extract features
    print "extracting training features..."
    X_train = pickle.load(open(train_data_file, 'rb'))
    t_train = pickle.load(open(train_classes_file, 'rb'))
    print "done extracting training features"
    print

    # train here, and learn your classification parameters
    print "learning..."

    best_num_trees = 128
    best_depth = 25

    # # Find best depth
    # scores = []
    # for i in range(1, 26):
    #   clf = RandomForestClassifier(max_depth=i)
    #   clf.fit(X_train, t_train)
    #   y_hat = clf.predict(X_train)
    #   # scores.append(clf.score(X_train, t_train))
    #   scores.append(r2_score(t_train, y_hat))
    # index = np.argmax(np.array(scores))
    # best_depth = index + 1
    # print "best_depth: {}".format(best_depth) 
    # print "best_score: {}".format(scores[index])

    # # Find best num_estimators
    # scores = []
    # num_trees = [2**x for x in range(8)]  # 2, 4, 8, 16, 32, ...
    # for i in num_trees:
    #   clf = RandomForestClassifier(n_estimators=i, max_depth=best_depth)
    #   clf.fit(X_train, t_train)
    #   y_hat = clf.predict(X_train)
    #   # scores.append(clf.score(X_train, t_train))
    #   scores.append(r2_score(t_train, y_hat))
    # index = np.argmax(np.array(scores))
    # best_num_trees = 2 ** index
    # print "best_num_trees: {}".format(best_num_trees) 
    # print "best_score: {}".format(scores[index])
    
    # accuracy decreases with class_weight=class_dist
    # class_dist = {0:.0369, 1:.0162, 2:.012, 3:.0103,4:.0133,5:.0126,6:.0172,
    # 7:.0133,8:.5214,9:.0068,10:.1756,11:.0104,12:.1218,13:.0191,14:.013}

    clf = RandomForestClassifier(n_estimators=best_num_trees, max_depth=best_depth)
    clf.fit(X_train, t_train)
    y_hat = clf.predict(X_train)
    print "score: {}".format(r2_score(t_train, y_hat))

    print "done learning"
    print

    make_predictions()

if __name__ == "__main__":
    main()
    