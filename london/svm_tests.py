from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
import dataproc as dp
import numpy as np
import utils


# Load the data
path = '/home/avasbr/Desktop/kaggle_competitions/london/dataset'
train_data_path = path+'/train.csv'
train_label_path = path+'/trainLabels.csv'
test_data_path = path+'/test.csv'

X_tr = dp.read_csv_file(train_data_path)
X_te = dp.read_csv_file(test_data_path)

y_tr = dp.read_csv_file(train_label_path)
m_tr,d = X_tr.shape
m_te = X_te.shape[0]

X_tr,(mu,s) = dp.normalize_range(X_tr,axis=0)
# C_range = 10.0**np.arange(-2,9)
# gamma_range = 10.0**np.arange(-5,4)
# cv = StratifiedKFold(y=y_tr,n_folds=5)
# param_grid = dict(gamma=gamma_range,C=C_range)
# grid = GridSearchCV(svm.SVC(),param_grid=param_grid,cv=cv)
# grid.fit(X_tr,y_tr)

# print(grid.best_estimator_)

cv_err = []
for idx,(tr_idx,val_idx) in enumerate(dp.cross_val_idx(m_tr)):
	clf = svm.SVC(C=1e6,gamma=0.001)
	clf.fit(X_tr[tr_idx,:],y_tr[tr_idx])
	pred = clf.predict(X_tr[val_idx,:])
	mce = 1.0-np.mean(pred==y_tr[val_idx])
	cv_err.append(mce)
	print 'Iteration',idx+1,'error:',100*mce,'%'
avg_err = 1.*sum(cv_err)/len(cv_err)
print 'Average Cross-validation Error:',100.*(avg_err),'%'

clf = svm.SVC(C=1e6,gamma=0.001)
X_te = (X_te - mu)/s
pred = clf.fit(X_tr,y_tr).predict(X_te)
utils.write_solution(path,'svm',m_te,pred)
