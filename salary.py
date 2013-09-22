import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

'''
load data
model data
explore model
export predictions
'''


'''
'''
def shuffleDataFrame(df):
	indices = range(len(df))
	np.random.shuffle(indices)
	return df.iloc[indices]


'''
'''
def makeNFold(df, n=5):
	nRows = len(df)
	folds = np.array(range(nRows))*n/nRows
	np.random.shuffle(folds)
	df['Folds'] = folds


'''
'''
def randomSplit(df, p=0.8):
	df = shuffleDataFrame(df)
	n = len(df)*p
	train = df[:n]
	test = df[n:]
	return train, test


'''
'''
def encodeCategorical(data):
	data = data.copy()
	data[data.isnull()] = 'nan'
	categories = data.unique()
	index = range(len(categories))

	for i in index: 
		data[data==categories[i]] = i

	enc = OneHotEncoder()
	enc.fit(np.matrix(index).transpose())
	data = enc.transform(np.matrix(data).transpose())
	return pd.DataFrame(data.toarray())




'''
M columns, each a variable. The ith variable has n_i categories.
Turn each column into integers.
Turn the data frame of integers into a matrix.
'''
def encodeMultiCategorical(data):
	data = data.copy()
	num_of_cols = len(data.columns)
	n = np.zeros(num_of_columns)

	# for each column
	for i in range(num_of_cols):
		col = data.columns[i]
		categories = data[col].unique()
		n[i] = len(categories)

	enc = OneHotEncoder()
	enc.fit(np.matrix(index).transpose())
	data = enc.transform(np.matrix(data).transpose())
	return pd.DataFrame(data.toarray())


if __name__ == "__main__":
	dataFileName = "train.csv"
	np.random.seed(0)

	data = pd.read_csv(dataFileName)
	X = encodeCategorical(data['ContractType'])
	data = pd.concat([data, X], axis=1)

	train, test = randomSplit(data)

# 	features = 'ContractType'
# #	features = ['SourceName', 'LocationNormalized', 'Company', 'ContractTime', 'ContractType']
# 	target = 'SalaryNormalized'

	regressors = train.ix[:,0:]
	target = train['SalaryNormalized']

	clf = LinearRegression()
	clf.fit(regressors, target)

	y_pred = clf.predict(test.ix[:,0:])
	y_true = test['SalaryNormalized']

	print('MAE:', metrics.mean_absolute_error(y_true, y_pred))
	print('MSE:', metrics.mean_squared_error(y_true, y_pred))
