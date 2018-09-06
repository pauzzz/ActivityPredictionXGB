import numpy as np
import xgboost as xgb

xtrain=np.loadtxt('train/X_train.txt', delimiter=' ')
ytrain=np.loadtxt('train/Y_train.txt')

xtrain.shape
ytrain.shape

xtest=np.loadtxt('test/X_test.txt', delimiter=' ')
ytest=np.loadtxt('test/Y_test.txt')

xtest.shape
ytest.shape



#xtrain=pd.DataFrame(xtrain)
#xtrain['labels']=ytrain.tolist()
#
#xtest=pd.DataFrame(xtest)
#xtest['labels']=ytest.tolist()

xgb_train=xgb.DMatrix(xtrain, label=ytrain)
xgb_test=xgb.DMatrix(xtest, label=ytest)


#YIKES tried to run this, it killed my computer. Try on a super computer :^)
##test cv params for best params, optimizing
#
#cv_params1={'max_depth':[3,5,7], 'min_child_weight':[1,3,5]}
#
#ind_params1={'learning_rate': 0.1 , 'n_estimators':500, 'seed':123, 'subsample':0.8, 'colsample_bytree':0.8, 'objective':'multi:softmax' }
#
#optimized_GBM= GridSearchCV(xgb.XGBClassifier(**ind_params1), cv_params1, scoring='accuracy', cv=3, n_jobs=-1)
#
##now run our gridsearch with cross validation to see which params perform best.
##should take a LONG time
#optimized_GBM.fit(xtrain, ytrain)
##output here:
#
#
#
##we use grid scores to determine best params.
#optimized_GBM.grid_scores_

#set up params for xgboost model

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


watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
num_round = 25
bst = xgb.train(params, xgb_train, num_round, watchlist)
# get prediction
pred = bst.predict(xgb_test)
error_rate = np.sum(pred != ytest) / ytest.shape[0]
print('Accuracy using softmax = {}'.format(1-error_rate))

#predicteds are 90.51% accurate.









#label_dict={1:'WALKING',
#2:'WALKING_UPSTAIRS',
#3:'WALKING_DOWNSTAIRS',
#4:'SITTING',
#5:'STANDING',
#6:'LAYING'
#}
#
#data['labels']=data['rawlabel'].map(label_dict)
