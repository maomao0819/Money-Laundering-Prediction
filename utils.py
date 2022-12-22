# !pip install xgboost==1.7.1

# import library
import os
import pandas as pd
import numpy as np
import time
import pickle
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

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

def write_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def preprocess_train(df_y, df_cus_info, df_date, cus_data, date_col, data_use_col):
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
    return training_data, labels

def preprocess_test_public(df_public_x, df_cus_info, cus_data, date_col, data_use_col):
    print('Start processing public testing data')
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
    return testing_data, testing_alert_key

def preprocess_test_private(private_keys, df_cus_info, cus_data, date_col, data_use_col):
    print('Start processing private testing data')
    testing_data, testing_alert_key = [], []
    print(df_cus_info.shape)
    for i, private_key in enumerate(private_keys):
        # from alert key to get customer information
        alert_key = private_key
        date = 393
        if len(df_cus_info[df_cus_info['alert_key']==alert_key]) == 0:
            continue
        print(i)
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
        print('\r processing data {}/{}'.format(i+1, len(private_keys)), end = '')
    print('bye')
    return testing_data, testing_alert_key


def preprocess(data_dir, preprocess_data_dir):
    # declare csv path

    # [alert_key, date]
    train_alert_date_csv = os.path.join(data_dir, 'train_x_alert_date.csv')
    # [alert_key, cust_id, risk_rank, occupation_code, total_asset, AGE]
    cus_info_csv = os.path.join(data_dir, 'public_train_x_custinfo_full_hashed.csv')
    # [alert_key, sar_flag]
    y_csv = os.path.join(data_dir, 'train_y_answer.csv')

    # [cust_id, lupay, byymm, cycam, usgam, clamt, csamt, inamt, cucsm, cucah]
    ccba_csv = os.path.join(data_dir, 'public_train_x_ccba_full_hashed.csv')
    # [cust_id, date, country, cur_type, amt]
    cdtx_csv = os.path.join(data_dir, 'public_train_x_cdtx0001_full_hashed.csv')
    # [cust_id, debit_credit, tx_date, tx_time, tx_type, tx_amt, exchg_rate, info_asset_code, fiscTxId, txbranch, cross_bank, ATM]
    dp_csv = os.path.join(data_dir, 'public_train_x_dp_full_hashed.csv')
    # [cust_id, trans_date, trans_no, trade_amount_usd]
    remit_csv = os.path.join(data_dir, 'public_train_x_remit1_full_hashed.csv')
    # [alert_key, date]
    public_x_csv = os.path.join(data_dir, 'public_x_alert_date.csv')
    # [alert_key, sar_flag]
    public_private_x_csv = os.path.join(data_dir, '預測的案件名單及提交檔案範例.csv')

    training_data_file = os.path.join(preprocess_data_dir, 'training_data.pickle')
    labels_file = os.path.join(preprocess_data_dir, 'labels.pickle')
    public_testing_data_file = os.path.join(preprocess_data_dir, 'public_testing_data.pickle')
    public_testing_alert_key_file = os.path.join(preprocess_data_dir, 'public_testing_alert_key.pickle')
    private_testing_data_file = os.path.join(preprocess_data_dir, 'private_testing_data.pickle')
    private_testing_alert_key_file = os.path.join(preprocess_data_dir, 'private_testing_alert_key.pickle')


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
    df_public_private_x = pd.read_csv(public_private_x_csv)

    # do label encoding
    le = LabelEncoder()
    cus_data[2].debit_credit = le.fit_transform(cus_data[2].debit_credit)
    
    if (os.path.exists(training_data_file) and os.path.exists(labels_file)):
        training_data = load_from_pickle(training_data_file)
        labels = load_from_pickle(labels_file)
    else:
        training_data, labels = preprocess_train(df_y, df_cus_info, df_date, cus_data, date_col, data_use_col)

    if (os.path.exists(public_testing_data_file) and os.path.exists(public_testing_alert_key_file)):
        public_testing_data = load_from_pickle(public_testing_data_file)
        public_testing_alert_key = load_from_pickle(public_testing_alert_key_file)
    else:
        public_testing_data, public_testing_alert_key = preprocess_test_public(df_public_x, df_cus_info, cus_data, date_col, data_use_col)

    if (os.path.exists(private_testing_data_file) and os.path.exists(private_testing_alert_key_file)):
        private_testing_data = load_from_pickle(private_testing_data_file)
        private_testing_alert_key = load_from_pickle(private_testing_alert_key_file)
    else:
        private_keys = list(set(df_public_private_x['alert_key']) - (set(df_public_x['alert_key'])))
        private_testing_data, private_testing_alert_key = preprocess_test_private(private_keys, df_cus_info, cus_data, date_col, data_use_col)
    
    return np.array(training_data), labels, np.array(public_testing_data), public_testing_alert_key, np.array(private_testing_data), private_testing_alert_key,



def generate_data(data_dir, preprocess_data_dir, training_data_file, labels_file, public_testing_data_file, public_testing_alert_key_file, private_testing_data_file, private_testing_alert_key_file):
    training_data, labels, public_testing_data, public_testing_alert_key, private_testing_data, private_testing_alert_key = preprocess(data_dir, preprocess_data_dir)
    write_to_pickle(training_data_file, training_data)
    write_to_pickle(labels_file, labels)
    write_to_pickle(public_testing_data_file, public_testing_data)
    write_to_pickle(public_testing_alert_key_file, public_testing_alert_key)
    write_to_pickle(private_testing_data_file, private_testing_data)
    write_to_pickle(private_testing_alert_key_file, private_testing_alert_key)
    return training_data, labels, public_testing_data, public_testing_alert_key, private_testing_data, private_testing_alert_key


def load_data(training_data_file, labels_file, public_testing_data_file, public_testing_alert_key_file, private_testing_data_file, private_testing_alert_key_file):
    training_data = load_from_pickle(training_data_file)
    labels = load_from_pickle(labels_file)
    public_testing_data = load_from_pickle(public_testing_data_file)
    public_testing_alert_key = load_from_pickle(public_testing_alert_key_file)
    private_testing_data = load_from_pickle(private_testing_data_file)
    private_testing_alert_key = load_from_pickle(private_testing_alert_key_file)
    return training_data, labels, public_testing_data, public_testing_alert_key, private_testing_data, private_testing_alert_key


def get_data(data_dir='./data', preprocess_data_dir='preprocess_data'):
    training_data_file = os.path.join(preprocess_data_dir, 'training_data.pickle')
    labels_file = os.path.join(preprocess_data_dir, 'labels.pickle')
    public_testing_data_file = os.path.join(preprocess_data_dir, 'public_testing_data.pickle')
    public_testing_alert_key_file = os.path.join(preprocess_data_dir, 'public_testing_alert_key.pickle')
    private_testing_data_file = os.path.join(preprocess_data_dir, 'private_testing_data.pickle')
    private_testing_alert_key_file = os.path.join(preprocess_data_dir, 'private_testing_alert_key.pickle')
    if (os.path.exists(training_data_file) and os.path.exists(labels_file) and os.path.exists(public_testing_data_file) and os.path.exists(public_testing_alert_key_file) and os.path.exists(private_testing_data_file) and os.path.exists(private_testing_alert_key_file)) != 1:
        return generate_data(data_dir, preprocess_data_dir, training_data_file, labels_file, public_testing_data_file, public_testing_alert_key_file, private_testing_data_file, private_testing_alert_key_file)
    else:
        return load_data(training_data_file, labels_file, public_testing_data_file, public_testing_alert_key_file, private_testing_data_file, private_testing_alert_key_file)


def missing_imputate(training_data, testing_data, numerical_index, non_numerical_index, labels, remove_threshold_ratio=0.4):
    """## 缺失值補漏
    可以發現有不少筆資料其實是有缺漏的，補上缺失值的方法有很多種，我們對於數值類資料補上中位數，對於類別類資料補上眾數。
    """

    ''' Missing Value Imputation '''
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')


    numerical_data = training_data[:, numerical_index]
    non_numerical_data = training_data[:, non_numerical_index]

    remove_feature = np.count_nonzero(pd.isna(training_data), axis=0) > int(np.shape(training_data)[0] * remove_threshold_ratio)
    # training_data = np.delete(training_data, np.where(remove_feature)[0], 1)
    numerical_index = np.setdiff1d(numerical_index, np.where(remove_feature)[0])
    non_numerical_index = np.setdiff1d(non_numerical_index, np.where(remove_feature)[0])

    remove_record = np.count_nonzero(pd.isna(training_data), axis=1) > int(np.shape(training_data)[1] * remove_threshold_ratio)
    training_data = np.delete(training_data, np.where(remove_record)[0], 0)
    labels = np.delete(labels, np.where(remove_record)[0])
    

    numerical_data = training_data[:, numerical_index]
    non_numerical_data = training_data[:, non_numerical_index]

    imp_median.fit(numerical_data)
    numerical_data = imp_median.transform(numerical_data)

    imp_most_frequent.fit(non_numerical_data)
    non_numerical_data = imp_most_frequent.transform(non_numerical_data)

    training_data = np.concatenate((non_numerical_data, numerical_data), axis=1)

    test_numerical_data = testing_data[:, numerical_index]
    test_non_numerical_data = testing_data[:, non_numerical_index]

    test_numerical_data = imp_median.transform(test_numerical_data)

    test_non_numerical_data = imp_most_frequent.transform(test_non_numerical_data)

    testing_data = np.concatenate((test_non_numerical_data, test_numerical_data), axis=1)

    non_numerical_index = list(range(len(non_numerical_index)))
    numerical_index = list(range(len(non_numerical_index), len(numerical_index) + len(non_numerical_index)))

    return training_data, testing_data, numerical_index, non_numerical_index, labels

def normalize(training_data, public_testing_data, numerical_index, non_numerical_index):
    
    train_numerical_data = training_data[:, numerical_index]
    train_non_numerical_data = training_data[:, non_numerical_index]
    normalized_feature = np.mean(train_numerical_data, axis=0) > 100
    normalized_feature_ids = np.where(normalized_feature)[0]
    train_numerical_data[:, normalized_feature_ids] = MinMaxScaler().fit_transform(train_numerical_data[:, normalized_feature_ids])
    training_data = np.concatenate((train_non_numerical_data, train_numerical_data), axis=1)

    test_numerical_data = public_testing_data[:, numerical_index]
    test_non_numerical_data = public_testing_data[:, non_numerical_index]
    normalized_feature = np.average(test_numerical_data, axis=0) > 100
    normalized_feature_ids = np.where(normalized_feature)[0]
    test_numerical_data[:, normalized_feature_ids] = MinMaxScaler().fit_transform(test_numerical_data[:, normalized_feature_ids])
    testing_data = np.concatenate((test_non_numerical_data, test_numerical_data), axis=1)
    return training_data, testing_data

def onehot_encoding(training_data, testing_data, one_hot_index):
    onehotencorder = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), one_hot_index)],
        remainder='passthrough'                     
    )
    onehotencorder.fit(training_data)
    training_data = onehotencorder.transform(training_data)
    testing_data = onehotencorder.transform(testing_data)

    return training_data, testing_data

def save_checkpoint(checkpoint_path, model):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    state = model.state_dict()
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)