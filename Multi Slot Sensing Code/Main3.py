from AlgorithmsMultiSlot3 import Classification
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
    file_path = (base_path/"../Multi Slot Sensing Code/ClassificationDataSNR.csv").resolve()
    dataset= pd.read_csv(file_path,header=None)
    SNR=dataset.iloc[:].values
    Slots=int(SNR[0])
    Samples=int(SNR[1])
    SNR=SNR[2:]
    SU=len(SNR)

    file_path= (base_path/"../Multi Slot Sensing Code/ClassificationDataTrainY.csv").resolve()
    dataset = pd.read_csv(file_path,header=None)
    y_train = dataset.iloc[:].values
           
    file_path = (base_path/"../Multi Slot Sensing Code/ClassificationDataTestY.csv").resolve()
    dataset = pd.read_csv(file_path,header=None)
    y_test = dataset.iloc[:].values
    
    file_path= (base_path/"../Multi Slot Sensing Code/ClassificationDataTrainX.csv").resolve()
    dataset = pd.read_csv(file_path,header=None)
    X_train=dataset.iloc[:,:].values

    file_path = (base_path/"../Multi Slot Sensing Code/ClassificationDataTestX.csv").resolve()
    dataset = pd.read_csv(file_path,header=None)
    X_test=dataset.iloc[:,:].values
    
    # print("X_train",X_train)
    # print("y_train",y_train)
    # print("X_test",X_test)
    # print("y_test",y_test)
    # y = y.reshape(len(y),1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    

    X_test_2=copy.deepcopy(X_test)
    NormSNR=[x/np.sum(SNR) for x in SNR]
    for i in range(SU):
        X_test_2[:,i*Slots:(i+1)*Slots]=X_test_2[:,i*Slots:(i+1)*Slots]*NormSNR[i]
    
    demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,Samples=Samples,SU=SU,X_test_2=X_test_2,SNR=SNR,Slots=Slots)
    
    
    file=[]
    file2=[]
    file2.append(demo.Logistic())
    file2.append(demo.MLP())
    file2.append(demo.NaiveBayes())
    file2.append(demo.LinearSVM())
    file2.append(demo.GaussianSVM())
    file2.append(demo.OR())
    file2.append(demo.AND())
    file2.append(demo.MRC())
    # file.append(demo.S1())
    # file.append(demo.S2())
    # file.append(demo.S3())
    file.append(demo.RandomForest())
    file.append(demo.KNN())
    file.append(demo.XGBoost())
    file.append(demo.CatBoost())
    file.append(demo.ADABoost())
    
    # file.append(demo.DecisionTree())
    file2.sort(key=lambda x:x[2],reverse=True)
    file.sort(key=lambda x:x[2],reverse=True)
    
    if(file!=[]):
        plt.rcParams['figure.figsize'] = [12, 8]
        for [fpr,tpr,auc,type,colour,marker,markevery] in file:
            plt.plot(fpr, tpr, color=colour,marker=marker,markevery=markevery,ms=5,linewidth=1,label='%0.4f %s' %(auc,type))
        plt.title('ROC Curve')
        plt.grid() 
        plt.xlim([0, 0.4])
        plt.ylim([0.5, 1])
        plt.xticks([x/100 for x in range(0,45,5)])
        plt.ylabel('Probability of Detection')
        plt.xlabel('Pobability of False Alarm')
        plt.legend(title="AUC Values",loc = 'best')
        plt.show()   
        plt.clf()
        
    if(file2!=[]):
        plt.rcParams['figure.figsize'] = [12, 8]
        for [fpr,tpr,auc,type,colour,marker,markevery] in file2:
            plt.plot(fpr, tpr, color=colour,marker=marker,markevery=markevery,ms=5,label='%0.4f %s' %(auc,type))
        plt.title('ROC Curve')
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