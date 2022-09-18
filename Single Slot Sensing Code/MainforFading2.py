from AlgorithmsforFading2 import Classification
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
base_path = Path(__file__).parent
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy

def C():
    ramge=[0,4,8,12,16,20,24]
    name=["Rayleigh","Rician","Nakagami m=0.5","Nakagami m=1","Nakagami m=1.5","Nakagami m=2","AWGN"]
    marker=["o","s","p","P","*","X","D"]
    color=["red","green","lightskyblue","aqua","blue","midnightblue","black"]
    file2=[]
    
    for i in range(7):
        file_path1 = (base_path/"../Single Slot Sensing Code/ClassificationDataTrain.csv").resolve()
        # dataset = pd.read_csv(file_path1)
        dataset = pd.read_csv(file_path1,header=None)
        X_train= dataset.iloc[:, ramge[i]:ramge[i]+3].values
        y_train = dataset.iloc[:, ramge[i]+3].values
        
        file_path2 = (base_path/"../Single Slot Sensing Code/ClassificationDataTest.csv").resolve()
        dataset = pd.read_csv(file_path2,header=None)
        X_test = dataset.iloc[:, ramge[i]:ramge[i]+3].values
        y_test = dataset.iloc[:, ramge[i]+3].values
        
        Samples=50
        # print("X_train",X_train)
        # print("y_train",y_train)
        # print("X_test",X_test)
        # print("y_test",y_test)
        # y = y.reshape(len(y),1)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
        
        SU=3
        
        demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,Samples=Samples,SU=SU, name=name[i],color=color[i],marker=marker[i])
        
        
        # file2.append(demo.Logistic())
        # file2.append(demo.LinearSVM())
        file2.append(demo.GaussianSVM())
        # file2.append(demo.MLP())
        # file2.append(demo.NaiveBayes())
        # file2.append(demo.OR())
        # file2.append(demo.AND())
        # file2.append(demo.MRC())
        # file.append(demo.S1())
        # file.append(demo.S2())
        # file.append(demo.S3())
        # file.append(demo.RandomForest())
        # file.append(demo.KNN())
        # file.append(demo.XGBoost())
        # file.append(demo.CatBoost())
        # file.append(demo.ADABoost())
        # file.append(demo.DecisionTree())
    file2.sort(key=lambda x:x[2],reverse=True)
        
    if(file2!=[]):
        plt.rcParams['figure.figsize'] = [9, 5]
        for [fpr,tpr,auc,type,colour,marker,markevery] in file2:
            plt.plot(fpr, tpr, color=colour,marker=marker,markevery=markevery,ms=5,label='%0.4f %s' %(auc,type))
        plt.title('ROC Curve for GaussianSVM Variance=1')
        plt.grid()
        plt.xlim([0, 0.4])
        plt.ylim([0.5, 1])
        plt.xticks([x/100 for x in range(0,45,5)])
        plt.ylabel('Probability of Detection')
        plt.xlabel('Pobability of False Alarm')
        plt.legend(title="AUC Values",loc = 'best')
        plt.show()
        plt.clf()

    

C()