import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn import metrics
import sklearn.linear_model as lm

import string
import os
#from scipy.sparse import hstack, csr_matrix

'''
load data
model data
explore model
export predictions
'''


def text_features(text):
	n_words = float(len(text.split(' ')))
	n_alphanumeric = float(len([ch for ch in text if ch not in exclude]))
	n_caps = float(len([ch for ch in text if ch.isupper()]))
	word_len = n_alphanumeric/n_words
	cap_ratio = n_caps/n_alphanumeric
	caps_present = cap_ratio != 0
	punc_ratio = len([ch for ch in text if ch in exclude])/n_words
	return [n_words, cap_ratio, punc_ratio, word_len, caps_present]


def load(dataFileName):
	data = pd.read_csv(dataFileName)
	data = data.fillna('none')
	return data


def encodeCategorical(df, order=1):
    f_array = np.array(df)
    n = np.shape(f_array)

    if len(n) > 1:
    	f_array = [string.join(r, sep=':') for r in f_array]
    	f_array = [string.replace(r, ' ', '') for r in f_array]
    	f_array = [string.replace(r, ':', ' ') for r in f_array]
    else:
    	f_array = [string.replace(r, ' ', '') for r in f_array]

    order = min(order, n[-1])
    vect = CountVectorizer(ngram_range=(order, order))
    f_array = vect.fit_transform(np.hstack(f_array))

    return vect, f_array

def meanAbsoluteError(ground_truth, predictions):
	diff = np.abs(ground_truth - predictions)
	return np.sum(diff)/np.size(diff)


if __name__ == "__main__":
	np.random.seed(42)
	data = load("train_100k.csv")

	features = data[['ContractType', 'ContractTime', 'Category']]
	vect, f_array = encodeCategorical(features, 1)

	target = data['SalaryNormalized']

	clf = lm.LinearRegression()
	mae_scorer = metrics.make_scorer(meanAbsoluteError, greater_is_better=False)
	
	mae_scores = cross_validation.cross_val_score(clf, f_array, target, cv=5, scoring=mae_scorer)
	mse_scores = cross_validation.cross_val_score(clf, f_array, target, cv=5, scoring='mean_squared_error')

	print('MAE:', round(np.mean(mae_scores), 0))
	print('MSE:', round(np.mean(mse_scores), 0))

	clf.fit(f_array, target)
	pred = clf.predict(f_array)
	results = pd.DataFrame({'names': vect.get_feature_names(), 'weights': clf.coef_})
	results = results.sort('weights', ascending=0)
