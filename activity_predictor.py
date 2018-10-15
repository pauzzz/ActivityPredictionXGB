import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ggplot import *
import numpy as np

xtrain=pd.read_csv('train/X_train.csv', sep=' ', header=None)
ytrain=pd.read_csv('train/Y_train.csv', header=None)

print(xtrain.shape)
print(xtrain.describe())
print(ytrain.shape)

xtest=pd.read_csv('test/X_test.csv', sep=' ', header=None)
ytest=pd.read_csv('test/Y_test.csv', header=None)

print(xtest.shape)
print(ytest.shape)
#TSNE and data vis
samples=pd.concat([xtrain,ytrain],axis=1)
tsne=TSNE(learning_rate=100)
transformed=tsne.fit_transform(samples)
xs=transformed[:,0]
ys=transformed[:,1]

arr=ytrain[0].tolist()
#arr=np.array(list(map(lambda x: d[x], arr)))
plt.figure(figsize=(12, 9))
# Put a nicer background color on the legend.
fig=plt.scatter(xs, ys, alpha=0.4, cmap=plt.cm.get_cmap("tab20", 12), c=arr)
#plt.legend(handles=[arr],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
cbar=plt.colorbar(label='Activity Type')
fig.figure.savefig('TSNE.png', dpi=600)
fig.figure.show()

#compare general non optimized params to optimized params

tree=XGBClassifier(objective = 'multiclass:softmax' , learning_rate=0.1 ,n_estimators=1000)

tree.fit(xtrain, ytrain)


pred_tree =  tree.predict(xtest)

print(accuracy_score(ytest,pred_tree))

#issues setting up classifier.
#test cv params for best params, optimizing

cv_params = {'n_estimators':[1000,1200],'max_depth':[3,5]}
clf_params={'objective':'multiclass:softmax','nthread':2,'max_depth':3,'n_estimators':1000,'subsample':0.5}

clf=xgb.XGBClassifier(clf_params)

model= GridSearchCV(clf, cv_params, scoring='accuracy', cv=3, n_jobs=3)



#now run our gridsearch with cross validation to see which params perform best.
#should take a LONG time
model.fit(xtrain, ytrain)
#output here:



#we use grid scores to determine best params.
model.grid_scores_

#params for this first optimization are max_depth:3, n_estimators:1000.
#now play with subsampling and learning rate params
cv_params2={'learning_rate': [0.1,0.01], 'subsample': [0.5,1]}
clf_params2={'n_estimators':1000, 'objective':'multiclass:softmax', 'max_depth':3, }
#set up params for xgboost model
clf2=XGBClassifier(clf_params2)
model2=GridSearchCV(clf2, cv_params2, scoring='accuracy', cv=3, n_jobs=3, return_train_score=True)

model2.fit(xtrain, ytrain)


params={}
params['objective'] = 'multi:softmax'
# scale weight of positive examples
params['eta'] = 0.1
params['max_depth'] = 7
params['silent'] = 1
params['nthread'] = 4
params['num_class'] = 13
params['subsample']=0.8
params['colsample_bytree']=0.7
params['min_child_weight']=3


xgb_train=xgb.DMatrix(xtrain, label=ytrain)
xgb_test=xgb.DMatrix(xtest, label=ytest)

watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
num_round = 25
bst = xgb.train(params, xgb_train, num_round, watchlist)
# get prediction
pred = bst.predict(xgb_test)
error_rate = np.sum(pred != ytest) / ytest.shape[0]
print('Accuracy using softmax = {}'.format(1-error_rate))

#predicteds are 90.51% accurate.

#October 4 new addition: Use keras and neural networks









