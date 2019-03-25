import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score

def make_predictions():
  outputfile = "sample_predictions.csv"  # feel free to change this or take it as an argument
  
  # get rid of training data and load test data
  del X_train
  del t_train
  print "extracting test features..."
  X_test = pickle.load(open('X_test.p', 'rb'))
  test_ids = pickle.load(open('test_ids.p', 'rb'))
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
    X_train = pickle.load(open('X_train.p', 'rb'))
    t_train = pickle.load(open('t_train.p', 'rb'))
    print "done extracting training features"
    print

    # TODO train here, and learn your classification parameters
    print "learning..."

    scores = []
    for i in range(1, 11):
      print i
      clf = KNeighborsClassifier(n_neighbors=i)
      clf.fit(X_train, t_train)
      y_hat = clf.predict(X_train)
      scores.append(r2_score(t_train, y_hat))
    
    index = np.argmax(np.array(scores))
    best_num_neighbors = index + 1
    print "best_num_neighbors: {}".format(best_num_neighbors)
    print "score: {}".format(scores[index]) 

    # best results so far:
    # best_num_neighbors: 1
    # score: 0.935819598033   

    print "done learning"
    print

    # make_predictions()

if __name__ == "__main__":
    main()
    
