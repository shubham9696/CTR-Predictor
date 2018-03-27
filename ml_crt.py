import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing,svm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import pickle
import re

style.use('ggplot')

df=pd.read_csv('data/example_crt_prediction.csv')
df.head(10)

df.CTR=df.CTR.str.replace('%',"").str.replace(',','.').astype('float')
df.Position=df.Position.str.replace(',','.').astype('float')
df['Position']=pd.to_numeric(df['Position'])
df['CTR']=pd.to_numeric(df['CTR'])
df.round(0)

df.dropna(inplace=True)
df.head()

df.corr()['CTR']

features=["Position","Impressions"]

target="CTR"

train=df.sample(frac=0.8)
test=df.loc[~df.index.isin(train.index)]

print ("Train rows: {}".format(len(train.index)))
print ("Test rows: {}".format(len(test.index)))

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score

def print_scores(scores):
    r=1
    for score in scores:
        print("Run: {} - Score: {}".format((r,score)))
        r+=1

Model1=LinearRegression()
Model1.fit(train[features],train[target])
prediction_score=Model1.score(test[features],test[target])
print("The score of prediction for LinearRegressionModel is: {}".format(prediction_score))

Model2=DecisionTreeRegressor()
Model2.fit(train[features],train[target])
prediction_score=Model2.score(test[features],test[target])
print("The score of prediction for DecisionTreeClassifierModel is: {}".format(prediction_score))

Model3=SVR()
Model3.fit(train[features],train[target])
prediction_score=Model3.score(test[features],test[target])
print("The score of prediction for Support Vector Regression is: {}".format(prediction_score))

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

pipeline=make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=200))
hyper_params={'randomforestregressor__max_features':['auto','sqrt','log2'],'randomforestregressor__max_depth':[5,3]}
Model4=GridSearchCV(pipeline,hyper_params,cv=5)
Model4.fit(train[features],train[target])
prediction_score=Model4.score(test[features],test[target])
print("The score of prediction for RandomForestRegressorModel is: {}".format(prediction_score))

pos=2
impression=200
data=[[pos,impression]]

df_predict=pd.DataFrame(data=data,index=[0],columns=features)

res=Model1.predict(df_predict)
print("LinearRegression predicted: {}% CTR".format(int(res[0])))


res=Model2.predict(df_predict)
print("DecisionTree predicted: {}% CTR".format(int(res[0])))


res=Model3.predict(df_predict)
print("Support Vector predicted: {}% CTR".format(int(res[0])))


res=Model4.predict(df_predict)
print("RandomForest predicted: {}% CTR".format(int(res[0])))

def plt_ctr_position(models,features,from_pos,to_pos,data):
    for model in models:
        prediction_x=[]
        prediction_y=[]
        for pos in range(from_pos,to_pos):
            df_predict=pd.DataFrame(data=data,index=[0],columns=features)
            prediction_x.append(pos)
            prediction_y.append(model.predict(df_predict)[0])
        prediction_x,prediction_y
        plt.plot(prediction_x,prediction_y)


plt_ctr_position([Model1, Model2, Model3, Model4], features, 1, 20, data)


