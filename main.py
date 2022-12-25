import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from parser import parse_args
import torch
import torchvision
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm
import xgboost as xgb
import pickle
import utils
from engine import model_prob, ML_model_prob, predict

def main(args):
    df_train, df_public, df_private = utils.get_preprocessed_data(args)
    df_train['alert_key'] = df_train['alert_key'].astype(int)
    df_public['alert_key'] = df_public['alert_key'].astype(int)
    df_private['alert_key'] = df_private['alert_key'].astype(int)
    predict(args, df_train, df_public, df_private, pred_type='public', load=False, load_to_train=False)
    # xgbrModel = xgb.XGBClassifier(learning_rate=0.1,
    #     n_estimators=1000,         # 樹的個數--1000棵樹建立xgboost
    #     max_depth=6,               # 樹的深度
    #     min_child_weight = 1,      # 葉子節點最小權重
    #     gamma=0.,                  # 懲罰項中葉子結點個數前的參數
    #     subsample=0.8,             # 隨機選擇80%樣本建立決策樹
    # #   objective='multi:softmax', # 指定損失函數
    #     scale_pos_weight=1,        # 解決樣本個數不平衡的問題
    #     random_state=0            # 隨機數
    # )

    # """# RandomForestClassifier 訓練"""
    # RFC = RandomForestClassifier(n_estimators = 100)

    # """# KNeighborsClassifier 訓練"""
    # KNN = KNeighborsClassifier(n_neighbors=3)

    # """# DecisionTreeClassifier 訓練"""
    # DT = DecisionTreeClassifier()
    
    # """# SVM 訓練"""
    # svc = svm.SVC(C=1.0, probability=True)
    # svr = svm.SVR(C=1.0, epsilon=0.2)

    # df_pred_xgbr = ML_model_prob(xgbrModel, df_train, df_public, label_column='sar_flag')
    # df_pred_RFC = ML_model_prob(RFC, df_train, df_public, label_column='sar_flag')
    # df_pred_KNN = ML_model_prob(KNN, df_train, df_public, label_column='sar_flag')
    # df_pred_DT = ML_model_prob(DT, df_train, df_public, label_column='sar_flag')
    # df_pred_SVC = ML_model_prob(svc, df_train, df_public, label_column='sar_flag')
    # df_pred_SVR = ML_model_prob(svr, df_train, df_public, label_column='label')
    # df_pred_dnn = model_prob(args, df_train, df_public, df_private, pred_type='public', load=False)

    # pred_prob = 10 * df_pred_xgbr['probability'] + df_pred_RFC['probability'] + df_pred_KNN['probability'] + \
    #     df_pred_DT['probability'] + df_pred_SVC['probability'] + 2 * df_pred_SVR['probability'] + 12 * df_pred_dnn['probability']
    
    # pred_data = {'alert_key': df_public['alert_key'],
    #         'probability': pred_prob
    # }

    # df_pred = pd.DataFrame(pred_data)
    # df_pred = df_pred.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
    # utils.pred_to_csv(args, df_pred)
    # utils.evaluate(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
