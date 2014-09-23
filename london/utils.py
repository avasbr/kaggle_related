import dataproc as dp
import numpy as np

def load_london_dataset(path):
	''' Loads the kaggle london dataset, given the path where the three files
	train.csv, trainLabels.csv, and test.csv are found '''

	# read in the data
	train_data_path = path+'/train.csv'
	train_label_path = path+'/trainLabels.csv'
	test_data_path = path+'/test.csv'

	X_tr = dp.read_csv_file(train_data_path).T
	X_te = dp.read_csv_file(test_data_path).T
	
	# X = dp.normalize_range(X) # normalize the range for everything
	targets = dp.read_csv_file(train_label_path)
	y_tr = np.zeros([2,targets.size])
	
	for idx,target in enumerate(targets):
		y_tr[target,idx] = 1

	d,m_tr = X_tr.shape
	m_te = X_te.shape[1]
	k = y_tr.shape[0]

	return X_tr,y_tr,m_tr,X_te,m_te,d,k

def load_london_dataset_alt(path):
	''' Alternative version of loading the data '''

	train_data_path = path+'/train.csv'
	train_label_path = path+'/trainLabels.csv'
	test_data_path = path+'/test.csv'

	X_tr = dp.read_csv_file(train_data_path)
	X_te = dp.read_csv_file(test_data_path)
	
	# X = dp.normalize_range(X) # normalize the range for everything
	y_tr = dp.read_csv_file(train_label_path)
	m_tr,d = X_tr.shape
	m_te = X_te.shape[0]

	return X_tr,y_tr,m_tr,X_te,m_te,d

def write_solution(path,tech_str,m_te,pred):
	'''Writes the solution to a file be submitted to kaggle '''
	
	sol_path = path+'/'+tech_str+'_testLabels.csv'
	header = "Id,Solution"
	idx = range(1,m_te+1)
	sol = np.hstack((np.array(idx)[:,np.newaxis],pred[:,np.newaxis]))
	np.savetxt(sol_path,sol,fmt='%i,%i',header=header,comments='')

