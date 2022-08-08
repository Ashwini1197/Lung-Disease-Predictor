# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:11:59 2019

@author: ashwini
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:16:56 2017

Logistic regression

@author: Ilaria
"""

# Data Preprocessing Template

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")


# Importing the dataset
dataset = pd.read_csv('Dset.csv')
X = dataset.iloc[:, [2,6]].values
y = dataset.iloc[:, 13].values

feature_name = ['FEV1','FVC']
# Taking care of missing da""ta

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:1 ])
X[:, 0:1] = imputer.transform(X[:, 0:1])
print("X=%s" % (X))

# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Feature Scaling

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
#print("X=%s" % (X_test))
X_test = sc_X.transform(X_test)
print("X=%s" % (X_test))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix	


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#fitting logistic regression to the training set
classifier = Sequential()
            
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
            
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
            
 
            #predict test set result
y_pred = classifier.predict(X_test)
            
            #making the confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
            
            #Xnew = np.array([[45, 5]])
Xnew=sc_X.transform([[4.1,2.46]])
            
            
print("X=%s" % (Xnew))
            
            
ynew=classifier.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
if(ynew[0]==0):
    predict = "asthma"
elif(ynew[0]==1):
    predict = "COPD"
elif(ynew[0]==2):
    predict = "emphysema"
            
from sklearn.model_selection import cross_val_score
           
            
            
fig = Figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)
lab=['Asthma','Copd','emphysema']
ax.set_xticklabels([''] + lab)
ax.set_yticklabels([''] + lab)
ax.set_xlabel('Predicted')
ax.set_ylabel('true')
            
num_projects=dataset.groupby('Gender').count()
plt.bar(num_projects.index.values, num_projects['Disease'])
plt.xlabel('gender')
plt.ylabel('disease predicted')
plt.show()
            
mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)
    
N = 2
asthmaMeans = (3,6)
COPDMeans = (2,6)
EMeans=(10,2)
            
ind = np.arange(N)    # the x locations for the groups
width = 0.25       # the width of the bars: can also be len(x) sequence
    
p1 = ax.bar(ind, asthmaMeans, width, color=(0.2588,0.4433,1.0))
p2 = ax.bar(ind, COPDMeans, width, color=(1.0,0.5,0.62),
                    bottom=asthmaMeans)
ax.set_ylabel('Disease')
ax.set_xlabel('Gender')
ax.set_title('Scores by disease and gender')
        
ax.set_xticks(ind + width/2.)
ax.set_yticks(np.arange(0,20,2))
ax.set_xticklabels(('male', 'female'))
    
cm = (confusion_matrix(y_test, y_pred))
print(cm)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                                             cv = 10)
accuracy=(accuracies.mean())
print(accuracy)
                #get the avarage to have a better idea of the overall model performance
mean=accuracies.mean()
                #calculate variance
var=accuracies.std()
                
            



#visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), 
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                     alpha = 0.75, cmap= ListedColormap(('red', 'green','blue')))
plt.ylim(X2.min(), X1.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                c= ListedColormap(('red', 'green','blue'))(i), label= j)
            
plt.title('randomforest (training set)')
plt.xlabel('FEV1')
plt.ylabel('FVC')
plt.show()
            
            
            #visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), 
                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                     alpha = 0.75, cmap= ListedColormap(('red', 'green','blue')))
plt.xlim(X1.min(), X2.max())
plt.ylim(X2.min(), X1.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                c= ListedColormap(('red', 'green','blue'))(i), label= j)
            
plt.title('randomforest (test set)')
plt.xlabel('FEV1')
plt.ylabel('FVC')
plt.show()
















