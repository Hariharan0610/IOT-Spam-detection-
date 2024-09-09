#======================= IMPORT LIBRARIES ==============================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import mean_squared_error 
warnings.filterwarnings("ignore")
import xgboost as xgb
import math
import matplotlib.pyplot as plt


import streamlit as st

#============================ BACKGROUND IMAGE  ==========================

import base64



st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"An efficient spam detection technique for iot devices"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('IOT-Banner-1024x614.jpg')


#==================== DATA SELECTION ================================

filee = st.file_uploader("Choose Dataset",['csv'])

if filee is None:
    
    st.text("Kindly Choose Dataset")

else:
    
    print("-------------------------------------------------------")
    print("                      DATA SELECTION                   ")
    print("-------------------------------------------------------")
    print()
    dataset = pd.read_csv("home.csv")
    print(dataset.head(20))
    print()
    
    
    st.write("-------------------------------------------------------")
    st.write("                      DATA SELECTION                   ")
    st.write("-------------------------------------------------------")
    print()
    dataset = pd.read_csv("home.csv")
    st.write(dataset.head(20))
    print()
    
    
    
    
    #==================== DATA PREPROCESSING ================================
    
    #=== CHECK MISSING VALUES ===
    
    print("-------------------------------------------------------")
    print("               BEFORE CHECKING MISSING VALUES          ")
    print("-------------------------------------------------------")
    print()
    print(dataset.isnull().sum())
    print()
    
    st.write("-------------------------------------------------------")
    st.write("               BEFORE CHECKING MISSING VALUES          ")
    st.write("-------------------------------------------------------")
    print()
    st.write(dataset.isnull().sum())
    print()
    
    print("-------------------------------------------------------")
    print("               AFTER CHECKING MISSING VALUES          ")
    print("-------------------------------------------------------")
    print()
    dataset=dataset.replace(np.nan,0)
    print(dataset.isnull().sum())
    print()
    
    
    st.write("-------------------------------------------------------")
    st.write("               AFTER CHECKING MISSING VALUES          ")
    st.write("-------------------------------------------------------")
    print()
    dataset=dataset.replace(np.nan,0)
    st.write(dataset.isnull().sum())
    print()
    
    #==== LABEL ENCODING ====
    
    from sklearn import preprocessing 
    label_encoder = preprocessing.LabelEncoder() 
    print("------------------------------------------------------")
    print("                  BEFORE LABEL ENCODING               ")
    print("------------------------------------------------------")
    print()
    print(dataset['icon'].head(10))
    
    st.write("------------------------------------------------------")
    st.write("                  BEFORE LABEL ENCODING               ")
    st.write("------------------------------------------------------")
    print()
    st.write(dataset['icon'].head(10))
    
    
    
    
    a = dataset['icon']#.unique()
    print("------------------------------------------------------")
    print("                  AFTER LABEL ENCODING                ")
    print("------------------------------------------------------")
    print()
    dataset['icon']= label_encoder.fit_transform(dataset['icon'].astype(str)) 
    dataset['summary']= label_encoder.fit_transform(dataset['summary'].astype(str)) 
    dataset['cloudCover']= label_encoder.fit_transform(dataset['cloudCover'].astype(str)) 
    
    print(dataset['icon'].head(10))
    
    st.write("------------------------------------------------------")
    st.write("                  AFTER LABEL ENCODING               ")
    st.write("------------------------------------------------------")
    print()
    dataset['icon']= label_encoder.fit_transform(dataset['icon'].astype(str)) 
    dataset['summary']= label_encoder.fit_transform(dataset['summary'].astype(str)) 
    dataset['cloudCover']= label_encoder.fit_transform(dataset['cloudCover'].astype(str)) 
    
    st.write(dataset['icon'].head(10))
    
    dataset.loc[dataset['use [kW]']>= 1, 'use [kW]'] = 1
    dataset.loc[dataset['use [kW]']<1, 'use [kW]'] = 0
    
    
    print("--------------------------------------------------------------")
    print("                  BEFORE DROP UNWANTED COLUMNS                ")
    print("--------------------------------------------------------------")
    print()
    
    print(dataset.shape[1])
    
    #drop unwanted columns 
    dataset.columns = [col.replace(' [kW]', '') for col in dataset.columns]
    # dataset.columns
    dataset['sum_Furnace'] = dataset[['Furnace 1','Furnace 2']].sum(axis=1)
    dataset['avg_Kitchen'] = dataset[['Kitchen 12','Kitchen 14','Kitchen 38']].mean(axis=1)
    dataset = dataset.drop(['Kitchen 12','Kitchen 14','Kitchen 38'], axis=1)
    dataset = dataset.drop(['Furnace 1','Furnace 2'], axis=1)
    # dataset.columns
    dataset = dataset.drop(['time'], axis=1)
    dataset.iloc[np.r_[0:5,-5:0]].iloc[:,0] 
    
    dataset = dataset.drop(columns=['House overall'])
    # dataset.shape
    # dataset['icon'].value_counts()
    # dataset['summary'].value_counts()
    dataset = dataset.drop(columns=['summary', 'icon'])
    # dataset.shape
    # dataset['cloudCover'].unique()
    # dataset[dataset['cloudCover']=='cloudCover'].shape
    # dataset['cloudCover'][56:60]
    dataset['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
    dataset['cloudCover'] = dataset['cloudCover'].astype('float')
    # dataset['cloudCover'].unique()
    
    print("--------------------------------------------------------------")
    print("                  AFTER DROP UNWANTED COLUMNS                ")
    print("--------------------------------------------------------------")
    print()
    
    print(dataset.shape[1])
    
    
    
    #============================= DATA SPLITTING ===================================
    
    
    x = dataset.drop('use',axis=1)
    y = dataset['use']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    
    print("---------------------------------------------------------")
    print("                        DATA SPLITTING                   ")
    print("---------------------------------------------------------")
    print()
    print("Total number of rows in dataset:", dataset.shape[0])
    print()
    print("Total number of rows in training data:", X_train.shape[0])
    print()
    print("Total number of rows in testing data:", X_test.shape[0])
    
    
    st.write("---------------------------------------------------------")
    st.write("                        DATA SPLITTING                   ")
    st.write("---------------------------------------------------------")
    print()
    st.write("Total number of rows in dataset:", dataset.shape[0])
    print()
    st.write("Total number of rows in training data:", X_train.shape[0])
    print()
    st.write("Total number of rows in testing data:", X_test.shape[0])
    
    
    
    #============================= FEATURE EXTRACTION =============================
    
    #=== PCA ===
    
    from sklearn.decomposition import PCA 
    
    pca = PCA(n_components = 15) 
    
    X_train = pca.fit_transform(X_train) 
    X_test = pca.transform(X_test) 
    
    print("---------------------------------------------------")
    print("       PRINCIPLE COMPONENT ANALYSIS                ")
    print("---------------------------------------------------")
    print()
    print(" The original features is :", x.shape[1])
    print()
    print(" The reduced feature is :",X_train.shape[1])
    print()
    
    
    st.write("---------------------------------------------------")
    st.write("       PRINCIPLE COMPONENT ANALYSIS                ")
    st.write("---------------------------------------------------")
    print()
    st.write(" The original features is :", x.shape[1])
    print()
    st.write(" The reduced feature is :",X_train.shape[1])
    print()
    #============================= CLASSIFICATION =============================
    
    #=== XGBOOST LINEAR ====
    
    data_dmatrix = xgb.DMatrix(data=x,label=y)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 10)
    
    xg_reg.fit(X_train,y_train)
    
    preds = xg_reg.predict(X_test)
    
    
    print("---------------------------------------------------")
    print("               XGBOOST LINEAR MODEL                ")
    print("---------------------------------------------------")
    print()
    
    
    st.write("---------------------------------------------------")
    st.write("               XGBOOST LINEAR MODEL                ")
    st.write("---------------------------------------------------")
    print()
    
    
    
    from sklearn import metrics
    MAE_xg=metrics.mean_absolute_error(y_test,preds)
    MSE_xg=mean_squared_error(y_test,preds)
    RMSE_xg = math.sqrt(MSE_xg)
    Acc_xg=100-MAE_xg
    
    
    print()
    print(" 1.Mean Squared Error      :",MSE_xg)
    print()
    print(" 2.Mean Absolute Error     :",MAE_xg)
    print()
    print(" 3.Root mean squared error:",RMSE_xg)
    print()
    print(" 4.Accuracy               :",Acc_xg)
    print()
    
    print()
    st.write(" 1.Mean Squared Error      :",MSE_xg)
    print()
    st.write(" 2.Mean Absolute Error     :",MAE_xg)
    print()
    st.write(" 3.Root mean squared error:",RMSE_xg)
    print()
    st.write(" 4.Accuracy               :",Acc_xg)
    print()
    
    
    # Prediction Graph
    
    plt.plot(preds[0:100])
    plt.title("Prediction Graph For Xgboost")
    # plt.savefig("g1.png")
    plt.show()
    
    
    st.image("g1.png")
    
    st.write("------------------------------------------------------")

    
    # Comparison Graph
    
    import seaborn as sns
    sns.barplot(x=['MSE','MAE','RMSE'],y=[MSE_xg,MAE_xg,RMSE_xg])
    plt.title("Comparison Graph For Xgboost")
    # plt.savefig("g2.png")
    plt.show()
    
    
    st.image("g2.png")
    
    #=== BGLM ====
    
    st.write("------------------------------------------------------")

    
    print("------------------------------------------------------")
    print("      BAYESIAN GENERALIZED LINEAR MODEL (BGLM)        ")
    print("------------------------------------------------------")
    print()
    
    
    st.write("------------------------------------------------------")
    st.write("      BAYESIAN GENERALIZED LINEAR MODEL (BGLM)        ")
    st.write("------------------------------------------------------")
    print()
    
    
    from sklearn import linear_model
    
    Bayesglm = linear_model.BayesianRidge()
    
    Bayesglm.fit(X_train,y_train)
    
    preds1=Bayesglm.predict(X_test)
    
    MAE_bglm=metrics.mean_absolute_error(y_test,preds1)
    
    MSE_bglm=mean_squared_error(y_test,preds1)
    RMSE_bglm = math.sqrt(MSE_bglm)
    
    Acc_bglm=100-MAE_bglm
    
    print()
    print(" 1.Mean Squared Error   :",MSE_bglm)
    print()
    print(" 2.Mean Absolute Error  :",MAE_bglm)
    print()
    print(" 3.Root mean squared error:",RMSE_bglm)
    print()
    print(" 4.Accuracy               :",Acc_bglm)
    print()
    
    print()
    st.write(" 1.Mean Squared Error   :",MSE_bglm)
    print()
    st.write(" 2.Mean Absolute Error  :",MAE_bglm)
    print()
    st.write(" 3.Root mean squared error:",RMSE_bglm)
    print()
    st.write(" 4.Accuracy               :",Acc_bglm)
    print()
    # Prediction Graph
    
    plt.plot(preds1[0:100])
    plt.title("Prediction Graph For BGLM")
    # plt.savefig("g3.png")
    plt.show()
    
    st.image("g3.png")
    
    st.write("------------------------------------------------------")

    # Comparison Graph
    
    import seaborn as sns
    sns.barplot(x=['MSE','MAE','RMSE'],y=[MSE_bglm,MAE_bglm,RMSE_bglm])
    plt.title("Comparison Graph For BGLM")
    # plt.savefig("g4.png")
    plt.show()
    
    st.image("g4.png")

    
    st.write("------------------------------------------------------")

    #=== HYBRID XGBOOST AND BGLM ====
    
    
    import xgboost as xg
    
    xgb_r = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123)
    
    xgb_r.fit(X_train,y_train)
    
    preds11=xgb_r.predict(X_train)
    
    # ---- BGLM
    
    
    Bayesglm = linear_model.BayesianRidge()
    
    Bayesglm.fit(X_train,preds11)
    
    preds11=Bayesglm.predict(X_test)
    
    
    MAE_hyb=metrics.mean_absolute_error(y_test,preds11)
    
    MSE_hyb  =mean_squared_error(y_test,preds11)
    
    RMSE_hyb = math.sqrt(MSE_hyb)
    
    Acc_hyb=100-MAE_hyb
    
    print("------------------------------------------------------")
    print("          Hybrid BGLM and  XGBOOST                   ")
    print("------------------------------------------------------")
    print()
    
    st.write("------------------------------------------------------")
    st.write("          Hybrid BGLM and  XGBOOST                   ")
    st.write("------------------------------------------------------")
    print()
    
    print()
    print(" 1.Mean Squared Error   :",MSE_hyb)
    print()
    print(" 2.Mean Absolute Error  :",MAE_hyb)
    print()
    print(" 3.Root mean squared error:",RMSE_hyb)
    print()
    print(" 4.Accuracy               :",Acc_hyb)
    print()
    

    print()
    st.write(" 1.Mean Squared Error   :",MSE_hyb)
    print()
    st.write(" 2.Mean Absolute Error  :",MAE_hyb)
    print()
    st.write(" 3.Root mean squared error:",RMSE_hyb)
    print()
    st.write(" 4.Accuracy               :",Acc_hyb)
    print()    
    # Prediction Graph
    
    plt.plot(preds[0:100])
    plt.title("Prediction Graph For Hybrid")
    # plt.savefig("hyb1.png")
    plt.show()
    
    st.image("hyb1.png")
    # Comparison Graph
    
    import seaborn as sns
    sns.barplot(x=['MSE','MAE','RMSE'],y=[MSE_xg,MAE_xg,RMSE_xg])
    plt.title("Comparison Graph For Hybrid")
    # plt.savefig("hyb.png")
    plt.show()
    
    st.image("hyb.png")

    
    # ======================= PREDICTION ============
    
    print("---------------------------------------")
    print("    Prediction       ")
    print("---------------------------------------")
    print()
    
    st.write("---------------------------------------")
    st.write("    Prediction       ")
    st.write("---------------------------------------")
    print()
    
    
    # 55048 to 55099 75986 to 75999 76041 to 85023(spam)
    
    #=== SPAM OR NON SPAM ===
    
    # pred_value=int(input("Enter the predicted index value (0 to 503910):"))
    
    pred_value= 55099
    
    pred_value = st.text_input("Enter the predicted index value (0 to 503910):")
    
    butt = st.button("Submit")
    print()
    
    if butt:
    
        if preds1[int(pred_value)]>=0 and preds1[int(pred_value)]<=0.5:
            st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:26px;">{"Identified - Non Spam"}</h1>', unsafe_allow_html=True)

            print("===========================")
            print("--------- Non Spam  -------")
            print("===========================")
        else:
            st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:26px;">{"Identified - Spam"}</h1>', unsafe_allow_html=True)

            print("===========================")
            print("---------  Spam  ----------")
            print("===========================") 


