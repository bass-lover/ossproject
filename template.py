#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def load_dataset(dataset_path):
	df = pd.read_csv(dataset_path)
	return df
	#To-Do: Implement this function



def dataset_stat(dataset_df):
	fc = len(dataset_df)
	zc = dataset_df.value_counts(0)
	oc = dataset_df.value_counts(1)
	return fc,zc,oc
	#To-Do: Implement this function

def split_dataset(dataset_df, testset_size):
	csize = dataset_df.shape[1]
	x = np.array(pd.DataFrame(dataset_df[:csize-1]))
	y = np.array(pd.DataFrame(dataset_df, columns=['target']))
	x_train = x[testset_size+1:,:]
	x_test = x[:testset_size,:]
	y_train = y[testset_size+1:,:]
	y_test = y[:testset_size,:]
	return x_train,x_test,y_train,y_test
	#To-Do: Implement this function

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	dtc = DecisionTreeClassifier()
	dtc = dtc.fit(x_train,y_train)
	dtp = dtc.predict(x_test)
	for i in range(len(y_test)):
		if y_test[i]==1:
			if y_test[i]==dtp[i]:
				tp+=1
			else:
				fn+=1
		else:
			if y_test[i] == dtp[i]:
				tn += 1
			else:
				fp += 1
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	accuracy = (tp+tn)/(tp+fp+fn+tn)
	return accuracy,precision,recall
	#To-Do: Implement this function

def random_forest_train_test(x_train, x_test, y_train, y_test):
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	rfc = RandomForestClassifier()
	rfc.fit(x_train,y_train)
	rfp = rfc.predict(x_test)
	for i in range(len(y_test)):
		if y_test[i]==1:
			if y_test[i]==rfp[i]:
				tp+=1
			else:
				fn+=1
		else:
			if y_test[i] == rfp[i]:
				tn += 1
			else:
				fp += 1
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	accuracy = (tp+tn)/(tp+fp+fn+tn)
	return accuracy,precision,recall
	#To-Do: Implement this function

def svm_train_test(x_train, x_test, y_train, y_test):
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	svmc = SVC(kernel="linear")
	svmc.fit(x_train,y_train)
	svmp=svmc.predict(x_test)
	for i in range(len(y_test)):
		if y_test[i]==1:
			if y_test[i]==svmp[i]:
				tp+=1
			else:
				fn+=1
		else:
			if y_test[i] == svmp[i]:
				tn += 1
			else:
				fp += 1
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	accuracy = (tp+tn)/(tp+fp+fn+tn)
	return accuracy,precision,recall
	#To-Do: Implement this function

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)