from Algorithms1 import Classification
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
    file_path1 = (base_path/"../Single Slot Sensing Code/ClassificationDataTrain.csv").resolve()
    dataset = pd.read_csv(file_path1,header=None)
    X_train= dataset.iloc[:, 0:-1].values
    y_train = dataset.iloc[:, -1].values
       
    file_path2 = (base_path/"../Single Slot Sensing Code/ClassificationDataTest.csv").resolve()
    dataset = pd.read_csv(file_path2,header=None)
    X_test = dataset.iloc[:, 0:-1].values
    y_test = dataset.iloc[:, -1].values
    
    file_path3 = (base_path/"../Single Slot Sensing Code/ClassificationDataSNR.csv").resolve()
    SNR= pd.read_csv(file_path3,header=None)   
    Samples=50
    # print("X_train",X_train)
    # print("y_train",y_train)
    # print("X_test",X_test)
    # print("y_test",y_test)
    # y = y.reshape(len(y),1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    
    SNR2=[]
    SU=len(SNR)

    X_test_2=copy.deepcopy(X_test)    
    for i in range(SU):
        SNR2.append(SNR[0][i])
    SNR=SNR2
    NormSNR=[x/np.sum(SNR) for x in SNR]
    for i in range(SU):
        X_test_2[:,i]=X_test_2[:,i]*NormSNR[i]
    
    demo=Classification(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,Samples=Samples,SU=SU,X_test_2=X_test_2,SNR=SNR)
    
    
    file=[]
    file2=[]
    # file2.append(demo.Logistic())
    # file2.append(demo.MLP())
    # file2.append(demo.NaiveBayes())
    # file2.append(demo.LinearSVM())
    file2.append(demo.GaussianSVM())
    file2.append(demo.OR())
    file2.append(demo.AND())
    file2.append(demo.MRC())
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
    file.sort(key=lambda x:x[2],reverse=True)
    
    if(file!=[]):
        plt.rcParams['figure.figsize'] = [9,5]
        for [fpr,tpr,auc,type,colour,marker,markevery] in file:
            plt.plot(fpr, tpr, color=colour,marker=marker,markevery=markevery,ms=5,linewidth=1,label='%0.4f %s' %(auc,type))
            plt.title('ROC Curve')
        plt.grid() 
        plt.xlim([0, 0.4])
        plt.ylim([0.4, 0.9])
        plt.xticks([x/100 for x in range(0,45,5)])
        plt.ylabel('Probability of Detection')
        plt.xlabel('Pobability of False Alarm')
        plt.legend(title="AUC Values",loc = 'best')
        plt.show()   
        plt.clf()
        
    if(file2!=[]):
        plt.rcParams['figure.figsize'] = [9, 5]
        for [fpr,tpr,auc,type,colour,marker,markevery] in file2:
            plt.plot(fpr, tpr, color=colour,marker=marker,markevery=markevery,ms=5,label='%0.4f %s' %(auc,type))
        plt.title('ROC Curve')
        plt.grid()
        plt.xlim([0.0, 0.3])
        plt.ylim([0.6, 0.9])
        plt.xticks([x/100 for x in range(0,45,5)])
        plt.ylabel('Probability of Detection')
        plt.xlabel('Pobability of False Alarm')
        plt.legend(title="AUC Values",loc = 'best')
        plt.show()
        plt.clf()

    
    # datasetsize=[3,4,5,6,7]
    # auclinearsvm=   [0.9083,0.9272,0.9376,0.9432,0.9511]
    # aucgaussiansvm= [0.9061,0.9285,0.9353,0.9438,0.9506]
    # aucmlp=         [0.9060,0.9251,0.9376,0.9424,0.9498]
    # auclogistic=    [0.9043,0.9248,0.9373,0.9428,0.9509]
    # aucmrc=         [0.8991,0.9165,0.9252,0.9315,0.9377]
    # aucor=          [0.8934,0.9149,0.9258,0.9351,0.9423]
    # aucnb =         [0.8922,0.9044,0.9250,0.9373,0.9431]
    # aucand=         [0.7065,0.7255,0.7357,0.7428,0.7582]
    
    # plt.rcParams['figure.figsize'] = [9, 5]
    
    # plt.plot(datasetsize, auclinearsvm, color='fuchsia',marker="X",markevery=1,ms=5,linewidth=1,label='LinearSVM')
    # plt.plot(datasetsize, aucgaussiansvm, color='black',marker=2,markevery=1,ms=5,linewidth=1,label='GaussianSVM')
    # plt.plot(datasetsize, aucmlp, color='darkgreen',marker="$c$",markevery=1,ms=5,linewidth=1,label='MLP')
    # plt.plot(datasetsize, auclogistic, color='gold',marker="o",markevery=1,ms=5,linewidth=1,label='Logistic')
    # plt.plot(datasetsize, aucmrc, color='lightcoral',marker="<",markevery=1,ms=5,linewidth=1,label='MRC')
    # plt.plot(datasetsize, aucor, color='darkred',marker="v",markevery=1,ms=5,linewidth=1,label='OR')
    # plt.plot(datasetsize, aucnb, color='purple',marker="*",markevery=1,ms=5,linewidth=1,label='NaiveBayes')
    # plt.plot(datasetsize, aucand, color='brown',marker=">",markevery=1,ms=5,linewidth=1,label='AND')
    
    # plt.title('AUC vs Sensing Time')
    # plt.grid()
    # plt.legend(loc = 'lower right')
    # plt.xlim([2, 8])
    # plt.ylim([0.70, 1])
    # plt.ylabel('AUC Value')
    # plt.xlabel('Sensing Time (microseconds)')
    # plt.show()
    
    
    ############################################################################################################################3
    
    # datasetsize=[2,3,4,5,6,7]
    # auclinearsvm=   [0.9161, 0.9376, 0.9569, 0.9687, 0.9766, 0.9840]
    # aucgaussiansvm= [0.9165, 0.9353, 0.9572, 0.9701, 0.9792, 0.9841]
    # aucmlp=         [0.9162, 0.9376, 0.9564, 0.9666, 0.9730, 0.9840]
    # auclogistic=    [0.9144, 0.9373, 0.9567, 0.9687, 0.9755, 0.9844]
    # aucmrc=         [0.9130, 0.9252, 0.9446, 0.9600, 0.9703, 0.9785]
    # aucor=          [0.9070, 0.9258, 0.9454, 0.9574, 0.9669, 0.9733]
    # aucnb =         [0.9020, 0.9250, 0.9490, 0.9618, 0.9749, 0.9791]
    # auccatboost=    [0.9048, 0.9218, 0.9481, 0.9612, 0.9726, 0.9801]
    
    # plt.rcParams['figure.figsize'] = [9,5]
    # plt.plot(datasetsize, auclinearsvm, color='fuchsia',marker="X",markevery=1,ms=5,linewidth=1,label='LinearSVM')
    # plt.plot(datasetsize, aucgaussiansvm, color='black',marker=2,markevery=1,ms=5,linewidth=1,label='GaussianSVM')
    # plt.plot(datasetsize, aucmlp, color='darkgreen',marker="$c$",markevery=1,ms=5,linewidth=1,label='MLP')
    # plt.plot(datasetsize, auclogistic, color='gold',marker="o",markevery=1,ms=5,linewidth=1,label='Logistic')
    # plt.plot(datasetsize, aucmrc, color='lightcoral',marker="<",markevery=1,ms=5,linewidth=1,label='MRC')
    # plt.plot(datasetsize, aucor, color='darkred',marker="v",markevery=1,ms=5,linewidth=1,label='OR')
    # plt.plot(datasetsize, aucnb, color='purple',marker="*",markevery=1,ms=5,linewidth=1,label='NaiveBayes')
    # plt.plot(datasetsize, auccatboost, color='pink',marker="d",markevery=1,ms=5,linewidth=1,label='CatBoost')
    
    # plt.title('AUC vs SU number')
    # plt.grid()
    # plt.legend(loc = 'best')
    # plt.xlim([1, 8])
    # plt.ylim([0.88, 1])
    # plt.ylabel('AUC Value')
    # plt.xlabel('Number')
    # plt.show()
    
    
          
C()