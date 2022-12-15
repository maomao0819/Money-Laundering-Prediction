

# !pip install xgboost==1.7.1

# import library
import os
import pandas as pd
import numpy as np
import time
import collections
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

"""## 資料前處理
這邊針對訓練資料和測試的資料作整理。
baseline主要會使用到的csv檔案如下:
  - public_train_custinfo_full_hashed.csv: 包含主要要判斷的alert key對應的幾項參數，顧客id, 風險等級, 職業, 行內總資產和年齡。
  - train_x_alert_date: 作為訓練資料的alert key以及發生日期，共23906筆。
  - public_x_alert_date: 作為公開測試集的alert key，格式同上共1845筆。
  - train_y_answer: 訓練資料alert key對應最後是否SAR。
  - 預測的案件名單及提交檔案範例: 用於生成預測結果

除此之外，還會使用到顧客資訊當作訓練資料:
  - public_train_x_ccba_full_hashed.csv
  - public_train_x_cdtx0001_full_hashed.csv
  - public_train_x_dp_full_hashed.csv
  - public_train_x_remit1_full_hashed.csv

前處理的方式包含:
  - 從 alert key 檢索出顧客資訊
  - 對非數值 feature 做 label encoding
  - 從顧客資訊中挑選適合的 features 當作訓練資料，這裡挑選離 alert date 最近的一筆顧客資訊當作 features
  - 統計 training data 缺失值數量
"""

def preprocess(data_dir):
    # declare csv path
    train_alert_date_csv = os.path.join(data_dir, 'train_x_alert_date.csv')
    cus_info_csv = os.path.join(data_dir, 'public_train_x_custinfo_full_hashed.csv')
    y_csv = os.path.join(data_dir, 'train_y_answer.csv')

    ccba_csv = os.path.join(data_dir, 'public_train_x_ccba_full_hashed.csv')
    cdtx_csv = os.path.join(data_dir, 'public_train_x_cdtx0001_full_hashed.csv')
    dp_csv = os.path.join(data_dir, 'public_train_x_dp_full_hashed.csv')
    remit_csv = os.path.join(data_dir, 'public_train_x_remit1_full_hashed.csv')

    public_x_csv = os.path.join(data_dir, 'public_x_alert_date.csv')

    cus_csv = [ccba_csv, cdtx_csv, dp_csv, remit_csv]
    date_col = ['byymm', 'date', 'tx_date', 'trans_date']
    data_use_col = [[1,3,4,5,6,7,8,9],[2,3,4],[1,4,5,6,7,8,9,10,11],[2,3]]
    
    print('Reading csv...')
    # read csv
    df_y = pd.read_csv(y_csv)
    df_cus_info = pd.read_csv(cus_info_csv)
    df_date = pd.read_csv(train_alert_date_csv)
    cus_data = [pd.read_csv(_x) for _x in cus_csv]
    df_public_x = pd.read_csv(public_x_csv)

    # do label encoding
    le = LabelEncoder()
    cus_data[2].debit_credit = le.fit_transform(cus_data[2].debit_credit)

    

    
    cnts = [0] * 4
    labels = []
    training_data = []

    print('Start processing training data...')
    start = time.time()
    for i in range(df_y.shape[0]):
        # from alert key to get customer information
        cur_data = df_y.iloc[i]
        alert_key, label = cur_data['alert_key'], cur_data['sar_flag']

        cus_info = df_cus_info[df_cus_info['alert_key']==alert_key].iloc[0]
        cus_id = cus_info['cust_id']
        cus_features = cus_info.values[2:]

        date = df_date[df_date['alert_key']==alert_key].iloc[0]['date']


        cnt = 0
        for item, df in enumerate(cus_data):
            cus_additional_info = df[df['cust_id']==cus_id]
            cus_additional_info = cus_additional_info[cus_additional_info[date_col[item]]<=date]

            if cus_additional_info.empty:
                cnts[item] += 1
                len_item = len(data_use_col[item])
                if item == 2:
                    len_item -= 1
                cus_features = np.concatenate((cus_features, [np.nan] * len_item), axis=0)
            else:
                cur_cus_feature = cus_additional_info.loc[cus_additional_info[date_col[item]].idxmax()]
                
                cur_cus_feature = cur_cus_feature.values[data_use_col[item]]
                # 處理 實際金額 = 匯率*金額
                if item == 2:
                    cur_cus_feature = np.concatenate((cur_cus_feature[:2], [cur_cus_feature[2]*cur_cus_feature[3]], cur_cus_feature[4:]), axis=0)
                cus_features = np.concatenate((cus_features, cur_cus_feature), axis=0)
        labels.append(label)
        training_data.append(cus_features)
        print('\r processing data {}/{}'.format(i+1, df_y.shape[0]), end = '')
    print('Processing time: {:.3f} secs'.format(time.time()-start))
    print('Missing value of 4 csvs:', cnts)


    print('Start processing testing data')
    testing_data, testing_alert_key = [], []
    for i in range(df_public_x.shape[0]):
        # from alert key to get customer information
        cur_data = df_public_x.iloc[i]
        alert_key, date = cur_data['alert_key'], cur_data['date']

        cus_info = df_cus_info[df_cus_info['alert_key']==alert_key].iloc[0]
        cus_id = cus_info['cust_id']
        cus_features = cus_info.values[2:]

        for item, df in enumerate(cus_data):
            cus_additional_info = df[df['cust_id']==cus_id]
            cus_additional_info = cus_additional_info[cus_additional_info[date_col[item]]<=date]

            if cus_additional_info.empty:
                len_item = len(data_use_col[item])
                if item == 2:
                    len_item -= 1
                cus_features = np.concatenate((cus_features, [np.nan] * len_item), axis=0)
            else:
                cur_cus_feature = cus_additional_info.loc[cus_additional_info[date_col[item]].idxmax()]
                cur_cus_feature = cur_cus_feature.values[data_use_col[item]]
                # 處理 實際金額 = 匯率*金額
                if item == 2:
                    cur_cus_feature = np.concatenate((cur_cus_feature[:2], [cur_cus_feature[2]*cur_cus_feature[3]], cur_cus_feature[4:]), axis=0)
                cus_features = np.concatenate((cus_features, cur_cus_feature), axis=0)

        testing_data.append(cus_features)
        testing_alert_key.append(alert_key)
        # print(cus_features)
        print('\r processing data {}/{}'.format(i+1, df_public_x.shape[0]), end = '')
    return np.array(training_data), labels, np.array(testing_data), testing_alert_key

"""# 訓練資料處理"""

data_dir = './data'
# data preprocessing
training_data, labels, testing_data, testing_alert_key = preprocess(data_dir)
print(training_data[0])
print(training_data.shape, testing_data.shape)

"""## 缺失值補漏
  可以發現有不少筆資料其實是有缺漏的，補上缺失值的方法有很多種，我們對於數值類資料補上中位數，對於類別類資料補上眾數。
"""

''' Missing Value Imputation '''
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# for numerical index we do imputation using median
numerical_index = [2,4,5,6,7,8,9,10,11,14,17,24]
# Otherwise we select the most frequent
non_numerical_index = [0,1,3,12,13,15,16,18,19,20,21,22,23]

numerical_data = training_data[:, numerical_index]
non_numerical_data = training_data[:, non_numerical_index]

imp_median.fit(numerical_data)
numerical_data = imp_median.transform(numerical_data)

imp_most_frequent.fit(non_numerical_data)
non_numerical_data = imp_most_frequent.transform(non_numerical_data)

training_data = np.concatenate((non_numerical_data, numerical_data), axis=1)

"""  此外，若類別類資料跟數字大小沒關係，我們採用 one-hot encoding 將其編碼。"""

# for some catogorical features, we do one hot encoding
one_hot_index = [1,3,4,5,6,7,8,9,12]
onehotencorder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), one_hot_index)],
    remainder='passthrough'                     
)
onehotencorder.fit(training_data)
training_data = onehotencorder.transform(training_data)
print(training_data.shape)

"""# XGBoost 訓練"""

import xgboost as xgb
# 建立 XGBClassifier 模型
xgbrModel=xgb.XGBClassifier(random_state=0)
# 使用訓練資料訓練模型
xgbrModel.fit(training_data, labels)

"""# 預測與結果輸出
  利用訓練好的模型對目標alert key預測報SAR的機率以及輸出為目標格式。
  目標輸出筆數3850，其中public筆數為1845筆。
  因上傳格式需要private跟public alert key皆考慮，直接從預測範本統計要預測的alert key，預測結果輸出為prediction.csv。
"""

# Do missing value imputation and one-hot encoding for testing data
test_numerical_data = testing_data[:, numerical_index]
test_non_numerical_data = testing_data[:, non_numerical_index]

test_numerical_data = imp_median.transform(test_numerical_data)

test_non_numerical_data = imp_most_frequent.transform(test_non_numerical_data)

testing_data = np.concatenate((test_non_numerical_data, test_numerical_data), axis=1)
testing_data = onehotencorder.transform(testing_data)

# Read csv of all alert keys need to be predicted
public_private_test_csv = os.path.join(data_dir, '預測的案件名單及提交檔案範例.csv')
df_public_private_test = pd.read_csv(public_private_test_csv)

# Predict probability
predicted = []
for i, _x in enumerate(xgbrModel.predict_proba(testing_data)):
    predicted.append([testing_alert_key[i], _x[1]])
predicted = sorted(predicted, reverse=True, key= lambda s: s[1])

# 考慮private alert key部分，滿足上傳條件
public_private_alert_key = df_public_private_test['alert_key'].values
print(len(public_private_alert_key))

# For alert key not in public, add zeros
for key in public_private_alert_key:
    if key not in testing_alert_key:
        predicted.append([key, 0])

predict_alert_key, predict_probability = [], []
for key, prob in predicted:
    predict_alert_key.append(key)
    predict_probability.append(prob)

df_predicted = pd.DataFrame({
    "alert_key": predict_alert_key,
    "probability": predict_probability
})

df_predicted.to_csv('prediction_baseline.csv', index=False)