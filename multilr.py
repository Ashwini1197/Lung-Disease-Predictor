# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:56:43 2019

@author: ashwini
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
dataset = pd.read_csv('mdata.csv',error_bad_lines=False)
df = pd.read_csv('mdata.csv', engine='python', encoding='utf_8_sig')

X = dataset.iloc[:, [0,2]].values
y = dataset.iloc[:, 11].values

# Taking care of missing da""ta


# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
#print("X=%s" % (X_test))
X_test = sc_X.transform(X_test)
print("X=%s" % (X_test))
# Feature Scaling
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix	
        
#fitting logistic regression to the training set
def LR(a,b):       
        
        from sklearn.linear_model import LogisticRegression
                    
        classifier = LogisticRegression(random_state = 0 )
        classifier.fit(X_train, y_train)
                    
                   
            #predict test set result
        y_pred = classifier.predict(X_test)
            
            #making the confusion matrix
                 
        cm = confusion_matrix(y_test, y_pred)
                    
            
            #Xnew = np.array([[45, 5]])
                    
        Xnew=sc_X.transform([[a,b]])
        print("X=%s" % (Xnew))
                    
            
            
        ynew=classifier.predict(Xnew)
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
        if(ynew[0]==0):
            predict ="Asthma"
        elif(ynew[0]==1):
            predict = "COPD"
        elif(ynew[0]==3):
            predict = "Restrictive"
        elif(ynew[0]==2):
            predict = "Normal"
            
        from sklearn.model_selection import cross_val_score
        cm = " Confusion matrix : " + str(confusion_matrix(y_test, y_pred))
                    
                    
        
                    
        
        
        
            
                
            # For Legend
                    
                    
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                                             cv = 10)
        accuracy=(accuracies.mean())
        accuracy=float("{0:.2f}".format(accuracy))
        
                #get the avarage to have a better idea of the overall model performance
        mean=accuracies.mean()
                #calculate variance
        var=accuracies.std()
                    
                
            
        #visualizing the training set results
        
        
        
        
        
        
        
        
        # Splitting the dataset into the Training set and Test set
        
        #visualizing the training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1, stop = X_set[:,0].max() + 1, step=0.01), 
                                         np.arange(start = X_set[:,1].min() - 1, stop = X_set[:,1].max() + 1, step=0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                                 alpha = 0.75, cmap= ListedColormap(('#ff6666','#99ff99','#66b3ff','#e4c542')))
        plt.xlim(X1.min(), X2.max())
        plt.ylim(X2.min(), X1.max())
        for i,j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set==j, 0], X_set[y_set==j, 1], 
                        c= ListedColormap(('red', 'green','blue','yellow'))(i), label= j)
                    
        plt.title('Logistic regression (training set)')
        plt.xlabel('FEV1')
        plt.ylabel('FVC')
        plt.savefig("C:/Users/ashwini/project/sem1/static/Lr1_train.png")
        
        
       
        #plt.legend()
        
        return (predict,cm,accuracy)
            
        

        


        

            
            














