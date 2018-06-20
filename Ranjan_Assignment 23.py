#
"""
Created on Sun Jun  3 14:23:28 2018

@author: ranjan
"""
#importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# loading dataset into a variabe
iris=load_iris()

#viewing feature-set and target details
print(iris.data)
print(iris.target)

#viewing shape of data
print(iris.data.shape)

# converting data into DataFrame to apply machine learning algo
irisDF=pd.DataFrame(iris.data)

#giving appropriate names to DataFrame columns
irisDF.columns=iris.feature_names
XData=irisDF

# viewing DataFrame and its detail
print(irisDF)
print(irisDF.describe())

#creating & viewing scatter_matrix to view relationships
scattermat=pd.scatter_matrix(irisDF,c=iris.target,figsize=(15,15),marker="o",hist_kwds={"bins":20},alpha=0.8)

#applyiing train_test_split functon to split the data
X_train,X_test,y_train,y_test=train_test_split(XData,iris.target,test_size=0.2,random_state=21,stratify=iris.target)

#selecting & applying scaling algo
scaler=MinMaxScaler()
X_train_scaled=scaler.fit(X_train).transform(X_train)
X_test_scaled=scaler.transform(X_test)

# selecting 3 clusters for PCA algorith
pca=PCA(n_components=3)
pca.fit(X_train_scaled,y_train)
X_pca=pca.transform(X_train_scaled)

ydf=pd.DataFrame(y_train)
ydf[0]=ydf[0].astype(float)

sn=ydf[0]

#viewing the initial and dimension-reduced shape
print(X_train_scaled.shape)
print(X_pca.shape)

xplot=np.array([])
yplot=np.array([])
zplot=np.array([])

fig=plt.figure()

ax=fig.add_subplot(111,projection="3d")


for k in range(X_pca.shape[0]):
    xplot=X_pca[k][0]
    yplot=X_pca[k][1]
    zplot=X_pca[k][2]
    ax.scatter(xplot,yplot,zplot,c=10,marker="o")
    
plt.show()





