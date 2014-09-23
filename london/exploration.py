import dataproc as dp
import kaggle_london_utils as klu
import numpy as np
import matplotlib.pyplot as plt

path='/home/avasbr/datasets/kaggle/london_dataset'
X_tr,y_tr,m_tr,X_te,m_te,d,k = klu.load_london_dataset(path)
X_tr = dp.normalize_range(X_tr) # normalize the range for everything

# look at individual feature distributions for classes
class_0_idx = np.where(y_tr[0]==1)[0]
class_1_idx = np.where(y_tr[1]==1)[0]
for var_idx in range(d):
	plt.subplot(8,5,var_idx)
	curr_feat_class_0 = X_tr[var_idx,class_0_idx]
	curr_feat_class_1 = X_tr[var_idx,class_1_idx]
	plt.hist(curr_feat_class_0,bins=30)
	plt.hist(curr_feat_class_1,bins=30)
plt.show()
