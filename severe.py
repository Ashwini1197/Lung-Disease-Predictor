import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


Dataset = pd.read_csv('mdata.csv')
#X = Dataset.iloc[:, [0,1]].values
#y = Dataset.iloc[:, 7].values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
from sklearn.preprocessing import LabelEncoder


asthma= Dataset[Dataset.Disease=="Asthma"]
copd= Dataset[Dataset.Disease=="COPD"]
emp= Dataset[Dataset.Disease=="Restrictive"]
normal = Dataset[Dataset.Disease=="Normal"]
def severe(a,ya,p,q):
    print(a)
    if(a=="Asthma"):
        x= asthma.iloc[:, [0,1]].values
        y= asthma.iloc[:, 10].values
    if(a=="COPD"):
        x= copd.iloc[:, [0,1]].values
        y= copd.iloc[:, 10].values
    if(a=="Restrictive"):
        x= emp.iloc[:, [0,1]].values
        y= emp.iloc[:, 10].values
    if(a=="Normal"):
        x= normal.iloc[:, [0,1]].values
        y= normal.iloc[:, 10].values
        
    x = sc_X.fit_transform(x)
    
# Encoding the Independent Variable
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)


    if(ya==1):
        classifier = LogisticRegression(random_state = 0 )
        classifier.fit(x,y)



        Xnew=sc_X.transform([[p,q]])
        ynew=classifier.predict(Xnew)
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))   

        if((ynew[0]==0) and a=="Normal") :
            predicted ="null"
        elif((ynew[0]==1) and a=="Asthma"):
            predicted = "mild"
        elif((ynew[0]==2) and a=="COPD"):
            predicted = "moderate"
        elif((ynew[0]==3) and a=="COPD"):
            predicted= "severe"

        print(predicted)	
        return (predicted)

                
                

    if(ya==2):
        classifier = GaussianNB()

        classifier.fit(x,y)



        Xnew=sc_X.transform([[p,q]])
        ynew=classifier.predict(Xnew)
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))   

        if((ynew[0]==0) and a=="Normal") :
            predicted ="null"
        elif((ynew[0]==1) and a=="Asthma"):
            predicted = "mild"
        elif((ynew[0]==2) and a=="COPD"):
            predicted = "moderate"
        elif((ynew[0]==3) and a=="COPD"):
            predicted= "severe"
        print(predicted)	
            
        return (predicted)

    if(ya==3):
        classifier = SVC(kernel='linear')

        classifier.fit(x,y)



        Xnew=sc_X.transform([[p,q]])
        ynew=classifier.predict(Xnew)   
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

        if((ynew[0]==0) and a=="Normal") :
            predicted ="null"
        elif((ynew[0]==1) and a=="Asthma"):
            predicted = "mild"
        elif((ynew[0]==2) and a=="COPD"):
            predicted = "moderate"
        elif((ynew[0]==3) and a=="COPD"):
            predicted= "severe"
        print(predicted)	
        return (predicted)


    if(ya==4):
        classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy',random_state=42)

        classifier.fit(x,y)



        Xnew=sc_X.transform([[p,q]])
        ynew=classifier.predict(Xnew)   
        print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

        if((ynew[0]==0) and a=="Asthma") :
            predicted ="mild"
        elif((ynew[0]==1) and a=="Asthma"):
            predicted = "moderate"
        elif((ynew[0]==0) and a=="COPD"):
            predicted = "moderate"
        elif((ynew[0]==1) and a=="COPD"):
            predicted= "severe"
        elif((ynew[0]==0) and a=="Normal"):
            predicted = "null"

        print(predicted)	
        return (predicted)            

