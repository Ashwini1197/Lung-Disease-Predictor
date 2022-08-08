# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:59:02 2019

@author: ashwini
"""

from multilr import LR
from svc import SVC
from temp2 import NB
from rf import RF
from severe import severe

from flask import Flask, render_template,request
app=Flask(__name__)
# Data Preprocessing Template


# prepare configuration for cross validation test harness

# prepare models






@app.route('/')
def index():
    return render_template('first2.html')


@app.route('/mainform1/')
def mainform1():
    return render_template('mainform1.html')


@app.route('/result',methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        res1 = request.form['fev1']
        res2=request.form['fvc']
        #res3 = request.form['age']
        print(res1)
        print(res2)

        models = []
        y_pred1=LR(res1,res2)
        y_pred2= NB(res1,res2)
        y_pred3= SVC(res1,res2)
        y_pred4= RF(res1,res2)
        models.append('LR')
        models.append('NB')
        models.append('SVM')
        models.append('RF')
# evaluate each model in turn

        if((y_pred1[2]>y_pred2[2]) and (y_pred1[2]>y_pred3[2]) and (y_pred1[2]>y_pred4[2])) :
            ya=1
        elif((y_pred2[2]>y_pred1[2]) and (y_pred2[2]>y_pred3[2]) and (y_pred2[2]>y_pred4[2])) :
            ya=2
        elif((y_pred3[2]>y_pred1[2]) and (y_pred3[2]>y_pred2[2]) and (y_pred3[2]>y_pred4[2])) :
            ya=3
        elif((y_pred4[2]>y_pred1[2]) and (y_pred4[2]>y_pred2[2]) and (y_pred4[2]>y_pred3[2])) :
            ya=4



        if(ya==1):
            sp=severe(y_pred1[0],ya,res1,res2)
        elif(ya==2):        
            sp=severe(y_pred2[0],ya,res1,res2)
        elif(ya==3):               
            sp=severe(y_pred3[0],ya,res1,res2)
        elif(ya==4):               
            sp=severe(y_pred4[0],ya,res1,res2)

        print(sp)     




        # render_template("ar.html",cm1=y_pred1[1],Accuracy1=y_pred1[2])
        return render_template("arr.html",cm1=y_pred1[1],Accuracy1=y_pred1[2],pred1 = y_pred1[0],cm2=y_pred2[1],Accuracy2=y_pred2[2],pred2 = y_pred2[0],cm3=y_pred3[1],
                              Accuracy3=y_pred3[2],pred3 = y_pred3[0],cm4=y_pred4[1],Accuracy4=y_pred4[2],pred4 = y_pred4[0],sev = sp)





if __name__ == '__main__':
    app.run()

