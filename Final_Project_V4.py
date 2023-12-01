from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

import pandas as pd
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt


    
def FP(Raw_Data_temp):
    epsilon = 0.8
    delta = 0.5
    
    learn_samples = 18
    
    X_Size = 50
    
    # np.random.shuffle(Raw_Data_temp)
    X = np.array(Raw_Data_temp.drop(columns=['comb08'])).copy()
    Years = np.array(Raw_Data_temp['year']).copy()
    Y = np.array(Raw_Data_temp['comb08']).copy()
    
    X_origin = np.array(Raw_Data_temp.drop(columns=['comb08'])).copy()
    
    
    X_Train2_Years = Years[X_Size:-1].copy()
    
    
    
    for i in range(len(X[1])):
        if(np.linalg.norm(X[:,i]) == 0):
            X[:,i] = X[:,i] / 1
        else:
            print(i)
            X[:,i] = X[:,i] / np.linalg.norm(X[:,i])
    
    
    
    X_Train = X[:X_Size].copy()
    X_Train2 = X[X_Size:-1].copy()
    X_origin = X_origin[X_Size:-1].copy()
    
    Y_Train = Y[:X_Size].copy()
    Y_Train2 = Y[X_Size:-1].copy()
    
    reg = LinearRegression().fit(X_Train, Y_Train)
    
    Titles = Raw_Data.drop(columns=['comb08','make','model']).columns.tolist()
    Coefficient = pd.DataFrame(reg.coef_).T
    Coefficient.columns=Titles
##    display(Coefficient)
    
    
    C = -7e3
    Q = (epsilon**2)/(X_Size*np.sqrt(np.log(1/(epsilon*delta)))*np.log(X_Size))
    
    
    alpha1 = C*Q*np.log(Q)
    alpha2 = epsilon/4
    
    print(alpha1,alpha2)
    print()
    
    print(reg.predict(X_Train2[0].reshape(1,-1)),Y_Train2[0],"\n")
    print("Training Started: \n")
    
    
    predict_error = []
    predict_year = []
    t = 0
    counter = 0
    while t<len(X_Train2):
        # print("\nt=",t)
        xt = X_Train2[t]
        U, S, Vh = LA.svd(np.matmul(X_Train.transpose(),X_Train), full_matrices=False)
        q = xt@LA.pinv(X_Train.transpose()@X_Train)@X_Train.transpose()
        u = U.transpose()@xt
        # print(LA.norm(q),alpha1,LA.norm(u),alpha2)
        if(LA.norm(q)<= alpha1 and LA.norm(u)<= alpha2):
            Y_predict = reg.predict(xt.reshape(1,-1))
            predict_error.append(Y_predict-Y_Train2[t])
            predict_year.append(int(X_Train2_Years[t]))
            # print(Y_predict,Y_Train2[t],"\n")
            t+=1
            # if(predict_error[-1]>100):
            #   print(X_origin[t],'\n',Y_predict,'\n',Y_Train2[t-1],'\n',t+X_Size)

        else:
            Y_predict = "[Reject]"
            # print(Y_predict,Y_Train2[t],"\n")
            for i in range(learn_samples):
                try:
                    yt = Y_Train2[t+i]
                    X_Train = np.delete(X_Train, 0,  axis=0)
                    Y_Train = np.delete(Y_Train, 0,  axis=0)

                    X_Train = np.append(X_Train, xt.reshape(1,-1), axis=0)
                    Y_Train = np.append(Y_Train, np.array([yt]), axis=0)
                except:
                    break
                reg = LinearRegression().fit(X_Train, Y_Train)
                t+=i
##                print(Y_predict)
        Titles = Raw_Data.drop(columns=['comb08','make','model']).columns.tolist()
        Coefficient = pd.DataFrame(reg.coef_).T
        Coefficient.columns=Titles
        # display(Coefficient)
        counter+=1
        # predict = [predict_year,predict_error]
        # plt.plot(predict_year,predict_error)
    predict_year.pop(np.argmax(predict_error))
    predict_error.pop(np.argmax(predict_error))
    
    
    error_list = [[],[]]
    
    error = 0
    count = 0
    
    for i in range(len(predict_error)-1):
        if(predict_year[i] == predict_year[i+1]):
            error+=np.abs(predict_error[i])
            count+=1
        else:
            try:
                error_list[1].append(float(error/count))
                error_list[0].append(int(predict_year[i]))
                error = 0
                count = 0
            except:
                error = 0
                count = 0
    
    VClass_Name = ['MiniCompact_Car','SubCompact_Car','Compact_Car','Standard_Car','Large_Car','Small_Station_Wagon','Standard_Station_Wagon','Large_Station_Wagon','Small_SUV','Standard_SUV','Minivan','Small_Truck','Standard_Truck','Van']
    plt.plot(error_list[0],error_list[1])
    plt.savefig('images\{:}_Mean_Error_by_Year.png'.format(VClass_Name[VClass_Num-1]))
    
    plt.plot(predict_error)
    plt.savefig('images\{:}_Error_by_index_Year.png'.format(VClass_Name[VClass_Num-1]))



Raw_Data = pd.read_csv('vehicles-processed-2.csv')
VClass_Num = 6

for Name in range(VClass_Num):
    
    # Raw_Data_temp = Raw_Data.drop(columns=['make','model','VClass','gears'])
    Raw_Data_temp = Raw_Data.drop(columns=['make','model'])
    Raw_Data_temp = Raw_Data_temp.loc[Raw_Data['VClass']==Name]
##    display(Raw_Data_temp)
    # Raw_Data_temp = np.array(Raw_Data_temp)
    
    
    # np.random.shuffle(Raw_Data_temp)
    X = np.array(Raw_Data_temp.drop(columns=['comb08'])).copy()
    Y = np.array(Raw_Data_temp['comb08']).copy()
    
    FP(Raw_Data_temp)
