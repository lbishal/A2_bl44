import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import scipy
np.random.seed(42) #set seed for randomness

def classify_movement(file_path):
  """ This method first extracts the movement features and lables, then uses naive bayes to classify the movements.
  """
  # Load the accelerometer file
  accelerometer_csv = np.loadtxt(file_path, delimiter=',')
  features, labels = extract_features_and_labels(accelerometer_csv)
  classification_accuracy(features, labels)


def extract_features_and_labels(raw_data):
  """ The method computes mean and variance for every 128 accelerometer data instances
  (i.e., for every 128 rows, you should get 3 mean and 3 variance values for the 3-axes of accelerometer data).
  In other words, we should get one 6 dimensional feature vector for each 128 data points.

  Args:
    raw_data:

  Returns:
    features:
    labels:
  """

  # Use time-window with length 128
  WINDOW_LENGTH = 128

  features = None
  labels = None

  assert raw_data is not None

  features, labels = [], []
  for i in range(0,len(raw_data)-WINDOW_LENGTH,WINDOW_LENGTH): #no overlap between windows
      #extract current snippet of data of WINDOW_LENGTH
      current_raw_acc     = raw_data[i:i+WINDOW_LENGTH,0:3]
      curr_activities     = raw_data[i:i+WINDOW_LENGTH,3]
      #compute mean and variaance of raw accelerometer as features
      curr_features = np.hstack([np.mean(current_raw_acc,axis=0),
                                 np.var(current_raw_acc,axis=0)])
      #take the mode of current activities as the label for the window
      curr_label    = scipy.stats.mode(curr_activities)[0]
      
      features.append(curr_features)
      labels.append(curr_label)

  features = np.array(features)
  labels = np.array(labels).squeeze()
  idx = np.arange(features.shape[0])
  np.random.shuffle(idx)
  features = features[idx]
  labels = labels[idx]

  return features, labels


def classification_accuracy(features, labels):
  """ Use Naive Bayes and cross-validation (supported by scikit-learn) to show the average accurcy for the classification.
  """
  ## TODO: Estimate the cross-validation accuracy for the classification using naive bayes.
  clf               = GaussianNB() #default scoring function of gaussianNB is accuracy
  num_folds         = 10
  accuracy_allfolds = cross_val_score(clf,X=features,
                             y=labels,cv=num_folds)
  assert(len(accuracy_allfolds) == num_folds)
  accuracy = accuracy_allfolds.mean()  
  print("Classification Accuracy:", accuracy)

if __name__ == "__main__":
  classify_movement("accelerometer_movement.csv")


