# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:59:02 2019

@author: ashwini
"""

from multilr import LR
from svc import SVC
from temp2 import NB
from rf import RF
from flask import Flask, render_template,request
app=Flask(__name__)
# Data Preprocessing Template


@app.route('/')
def index():
    return render_template('mainform1.html')


@app.route('/result',methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        res = request.form['type']
        res1 = request.form['fev1']
        res2=request.form['fvc']
      
        print(res)
        print(res1)
        print(res2)
        
        if(res=='lr'):
            y_pred=LR(res1,res2)
            return render_template("one.html",cm=y_pred[1],Accuracy=y_pred[2],Prediction=y_pred[0])
          
        if(res=='svm'):
            y_pred=SVC(res1,res2)
            return render_template("two.html",cm=y_pred[1],Accuracy=y_pred[2],Prediction=y_pred[0])

        if(res=='nb'):
            y_pred=NB(res1,res2)
            return render_template("three.html",cm=y_pred[1],Accuracy=y_pred[2],Prediction=y_pred[0])

        if(res=='rf'):
            y_pred=RF(res1,res2)
            return render_template("four.html",cm=y_pred[1],Accuracy=y_pred[2],Prediction=y_pred[0])
        
            
    

if __name__ == '__main__':
    app.run()
    
