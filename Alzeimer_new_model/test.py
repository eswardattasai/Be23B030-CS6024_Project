import os
import pandas as pd
import math as math
import numpy as np
import time
import itertools
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import *
from sklearn.svm import SVC

from tensorflow.keras.models import load_model


path= os.path.dirname(os.path.abspath(__file__))

#Read first independent testing file GSE109887
test_file_1=pd.read_csv(path+'/Data/Test_data1.csv')
#Read second independent testing file GSE138260
test_file_2=pd.read_csv(path+'/Data/Test_data2.csv')

#Load the pre-trained model
model = load_model(path + '/models/DNN41Genes.h5')

#Genes set to map the test files
genes_set=['ACO2', 'ADRA1B', 'CDC25A', 'COLEC12', 'COPE', 'DLGAP2',
	   'DNASE1L2', 'DUSP6', 'E2F5', 'FABP3', 'FBXO38', 'FCER1G', 'FDFT1',
	   'FOXD1', 'FUT6', 'KIFC3', 'LTF', 'LTK', 'MAP1B', 'MED14', 'MKRN3',
	   'MYRF', 'NEU3', 'NRIP2', 'NT5M', 'PAX5', 'PDE6H', 'PNMA3', 'PNOC',
	   'PPIG', 'PTAFR', 'RHOQ', 'SEMA6A', 'SLC12A9', 'SLC25A46', 'SST',
	   'TMF1', 'UNC13A', 'VCAM1', 'ZNF274', 'ZNF639','Label']



def features_selection(df1):
	df1=df1[genes_set]
	return df1


def predict_dnn(df1):
	df1=features_selection(df1)
	X2 = df1.drop('Label', axis=1)
	Y2= df1['Label'].values.astype(int)

	zeros=np.count_nonzero(Y2==0)
	ones=np.count_nonzero(Y2)
	
	#Apply log2 tranformaion
	X2=np.log2(X2)
	
	y_predict_score = model.predict(X2)
	auc_score=round(roc_auc_score(Y2, y_predict_score),4)
	
	#Convert floate scores into integer classes
	prediction=(y_predict_score > 0.5).astype("int32")
	prediction=list(itertools.chain.from_iterable(prediction.tolist()))
	y_pred=np.array(prediction)

	F1         = round ( f1_score(Y2, y_pred, average='weighted'),4)
	Recall     = round( recall_score(Y2, y_pred, average='weighted'),4 )
	Precision  = round( precision_score(Y2,y_pred, average='weighted'),4)

	print('AUC: ',auc_score)
	print('F1_score: ', F1)
	print('Recall: ', Recall)
	print('Precision: ',Precision)
	print('Number of samples in this testing file:{}, with {} ADs and {} controls'.format(len(df1),ones,zeros))





#-------------Print the results--------------------#
print('\nprediction of test file 1:')
predict_dnn(test_file_1)
print('\n------------------------\n')
print('prediction of test file 2:')
predict_dnn(test_file_2)
