import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
import pickle

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

def load_cus_df(args, split='public_train'):
    # [cust_id, lupay, byymm, cycam, usgam, clamt, csamt, inamt, cucsm, cucah]
    ccba_csv = os.path.join(args.data_dir, f'{split}_x_ccba_full_hashed.csv')
    # [cust_id, date, country, cur_type, amt]
    cdtx_csv = os.path.join(args.data_dir, f'{split}_x_cdtx0001_full_hashed.csv')
    # [cust_id, debit_credit, tx_date, tx_time, tx_type, tx_amt, exchg_rate, info_asset_code, fiscTxId, txbranch, cross_bank, ATM]
    dp_csv = os.path.join(args.data_dir, f'{split}_x_dp_full_hashed.csv')
    # [cust_id, trans_date, trans_no, trade_amount_usd]
    remit_csv = os.path.join(args.data_dir, f'{split}_x_remit1_full_hashed.csv')
    cus_csv = [ccba_csv, cdtx_csv, dp_csv, remit_csv]
    cus_data = [pd.read_csv(_x) for _x in cus_csv]
    cus_data[1].rename(columns = {'date': 'c_date'}, inplace = True)
    return cus_data

def concat_cus_data(args, df_data, cus_data):
    date_col = ['byymm', 'c_date', 'tx_date', 'trans_date']
    df_data = df_data.assign(label=0)
    df_data_ids = list(range(len(df_data)))
    df_data_batch_ids = [df_data_ids[i:i+args.df_batch_size] for i in range(0, len(df_data), args.df_batch_size)]
    df = pd.DataFrame()
    n_batch = len(df_data_batch_ids)
    batch_pbar = tqdm((df_data_batch_ids), total=n_batch, desc="Data Batch")

    for df_data_batch_id in batch_pbar:
        df_data_batch = df_data.iloc[df_data_batch_id]
        total_alert_keys = df_data_batch['alert_key'].unique()
        for df_cus_id, df_cus in enumerate(cus_data):
            df_data_batch = df_data_batch.merge(df_cus, on=['cust_id'])
            df_data_batch['day_diff'] = (df_data_batch[date_col[df_cus_id]] - df_data_batch['date']).abs()
            df_data_batch_closest = df_data_batch.loc[df_data_batch.groupby('alert_key')['day_diff'].idxmin()]
            df_data_batch_closest = df_data_batch_closest.drop(columns=['day_diff'])
            df_data_batch = df_data_batch.drop(columns=['day_diff'])
            # remove date > date + args.n_day_range
            df_data_batch = df_data_batch[(df_data_batch['date'] + args.n_day_range >= df_data_batch[date_col[df_cus_id]])]
            # remove date < date + args.n_day_range
            df_data_batch = df_data_batch[(df_data_batch['date'] - args.n_day_range <= df_data_batch[date_col[df_cus_id]])]
            df_data_batch = pd.concat([df_data_batch, df_data_batch_closest])
            # imputate nan alert keys
            imputate_alert_keys = list(set(total_alert_keys) - set(df_data_batch['alert_key'].unique()))
            imputate_index = pd.RangeIndex(len(imputate_alert_keys))
            df_imputate = pd.DataFrame(np.nan, index=imputate_index, columns=df_data_batch.columns)
            df_imputate['alert_key'] = imputate_alert_keys
            df_data_batch = pd.concat([df_data_batch, df_imputate])
            # label
            df_data_batch['label'] += ((df_data_batch[date_col[df_cus_id]] - df_data_batch['date']).abs() / args.n_day_range).clip(0, 1)
            df_data_batch = df_data_batch.drop(columns=[date_col[df_cus_id]])
            if df_cus_id == 2:
                df_data_batch['tx_amt'] = df_data_batch['tx_amt'] * df_data_batch['exchg_rate']
                df_data_batch = df_data_batch.drop(columns=['exchg_rate'])
            # reset 
            df_data_batch = df_data_batch.drop_duplicates()
            df_data_batch = df_data_batch.reset_index(drop=True)
        df = pd.concat([df, df_data_batch])
        batch_pbar.set_postfix(shape=df.shape)
    if 'sar_flag' in df.columns:
        df['label'] = (df['sar_flag'] * (1 - df['label'] / len(cus_data) / 2)).clip(0, 1)
    # reset
    df['alert_key'] = df['alert_key'].astype(int)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def generate_train_data(args):

    # [alert_key, date]
    train_alert_date_csv = os.path.join(args.data_dir, 'train_x_alert_date.csv')

    # [alert_key, cust_id, risk_rank, occupation_code, total_asset, AGE]
    cus_info_csv = os.path.join(args.data_dir, 'public_train_x_custinfo_full_hashed.csv')
    # [alert_key, sar_flag]
    y_csv = os.path.join(args.data_dir, 'train_y_answer.csv')

    df_y = pd.read_csv(y_csv)
    df_cus_info = pd.read_csv(cus_info_csv)
    df_date = pd.read_csv(train_alert_date_csv)
    cus_data = load_cus_df(args)

    df_data = df_y.merge(df_cus_info)
    df_data = df_data.merge(df_date)
    
    df_data = concat_cus_data(args, df_data, cus_data)

    write_to_pickle(args.train_pickle, df_data)
    return df_data

def generate_public_data(args):

    # [alert_key, date]
    public_alert_date_csv = os.path.join(args.data_dir, 'public_x_alert_date.csv')
    # [alert_key, cust_id, risk_rank, occupation_code, total_asset, AGE]
    cus_info_csv = os.path.join(args.data_dir, 'public_train_x_custinfo_full_hashed.csv')
    # [alert_key, sar_flag]
    y_csv = os.path.join(args.data_dir, '24_ESun_public_y_answer.csv')

    df_y = pd.read_csv(y_csv)
    df_cus_info = pd.read_csv(cus_info_csv)
    df_date = pd.read_csv(public_alert_date_csv)
    cus_data = load_cus_df(args)

    df_data = df_y.merge(df_cus_info)
    df_data = df_data.merge(df_date)

    df_data = concat_cus_data(args, df_data, cus_data)
    write_to_pickle(args.public_pickle, df_data)
    return df_data

def generate_private_data(args):

    # [alert_key, date]
    private_alert_date_csv = os.path.join(args.data_dir, 'private_x_alert_date.csv')
    # [alert_key, cust_id, risk_rank, occupation_code, total_asset, AGE]
    cus_info_csv = os.path.join(args.data_dir, 'private_x_custinfo_full_hashed.csv')

    df_cus_info = pd.read_csv(cus_info_csv)
    df_date = pd.read_csv(private_alert_date_csv)
    cus_data = load_cus_df(args, split='private')

    df_data = df_date.merge(df_cus_info)

    df_data = concat_cus_data(args, df_data, cus_data)
    write_to_pickle(args.private_pickle, df_data)
    return df_data

def get_train_data(args):
    if os.path.exists(args.train_pickle):
        df_train = load_from_pickle(args.train_pickle)
    else:
        df_train = generate_train_data(args)
    return df_train

def get_public_data(args):
    if os.path.exists(args.public_pickle):
        df_public = load_from_pickle(args.public_pickle)
    else:
        df_public = generate_public_data(args)
    return df_public

def get_private_data(args):
    if os.path.exists(args.private_pickle):
        df_private = load_from_pickle(args.private_pickle)
    else:
        df_private = generate_private_data(args)
    return df_private

def get_column_type(df_data):
    categorical_column = ['byymm', 'date', 'country', 'cur_type', 'risk_rank', 'occupation_code', 'AGE', 'debit_credit', 'tx_date', 'tx_time', 'tx_type', 'info_asset_code', 'fiscTxId', 'txbranch', 'cross_bank', 'ATM', 'trans_date', 'trans_no', 'sar_flag']
    no_process_column = ['alert_key', 'sar_flag', 'cust_id', 'label']
    all_column = set(df_data.columns.tolist())
    categorical_column = set(categorical_column)
    no_process_column = set(no_process_column)
    numerical_column = all_column - categorical_column - no_process_column
    categorical_column = all_column - numerical_column - no_process_column
    numerical_column = list(numerical_column)
    categorical_column = list(categorical_column)
    return numerical_column, categorical_column

def missing_remove(df_train, df_public, df_private, remove_threshold_ratio=0.6):
    keep_columns = df_train.notnull().sum(axis = 0) > len(df_train) * remove_threshold_ratio
    df_train = df_train[df_train.columns[keep_columns]]
    keep_rows = df_train.notnull().sum(axis = 1) > len(df_train.columns) * remove_threshold_ratio
    df_train = df_train[keep_rows]
    df_train = df_train.drop_duplicates()
    df_train = df_train.reset_index(drop=True)
    df_public = df_public[df_train.columns]
    df_private = df_private[df_train.columns]
    return df_train, df_public, df_private

def missing_imputate(df_train, df_public, df_private):
    """## 缺失值補漏
    可以發現有不少筆資料其實是有缺漏的，補上缺失值的方法有很多種，我們對於數值類資料補上中位數，對於類別類資料補上眾數。
    """

    ''' Missing Value Imputation '''
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imp_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    numerical_columns, categorical_columns = get_column_type(df_train)
    for numerical_column in numerical_columns:
        df_train[[numerical_column]] = imp_median.fit_transform(df_train[[numerical_column]])
        df_public[[numerical_column]] = pd.DataFrame(imp_median.transform(df_public[[numerical_column]].copy()))
        df_private[[numerical_column]] = pd.DataFrame(imp_median.transform(df_private[[numerical_column]].copy()))
    categorical_columns.append('label')
    categorical_columns.append('sar_flag')
    for categorical_column in categorical_columns:
        df_train[[categorical_column]] = imp_most_frequent.fit_transform(df_train[[categorical_column]])
        df_public[[categorical_column]] = pd.DataFrame(imp_most_frequent.transform(df_public[[categorical_column]].copy()))
        df_private[[categorical_column]] = pd.DataFrame(imp_most_frequent.transform(df_private[[categorical_column]].copy()))
    return df_train, df_public, df_private

def missing_process(df_train, df_public, df_private, remove_threshold_ratio=0.9):
    print('missing processing')
    df_train, df_public, df_private = missing_remove(df_train, df_public, df_private, remove_threshold_ratio=remove_threshold_ratio)
    df_train, df_public, df_private = missing_imputate(df_train, df_public, df_private)
    
    return df_train, df_public, df_private

def normalize(df_train, df_public, df_private, numerical_columns):
    print('normalizing')
    minmax = MinMaxScaler()
    df_train_numerical = df_train[numerical_columns].copy()
    df_public_numerical = df_public[numerical_columns].copy()
    df_private_numerical = df_private[numerical_columns].copy()
    normalized_columns = df_train_numerical.mean(axis=0) > 100
    normalized_columns = df_train_numerical.columns[normalized_columns]
    for normalized_column in normalized_columns:
        df_train_numerical[[normalized_column]] = pd.DataFrame(minmax.fit_transform(df_train_numerical[[normalized_column]].copy()))
        df_public_numerical[[normalized_column]] = pd.DataFrame(minmax.transform(df_public_numerical[[normalized_column]].copy()))
        df_private_numerical[[normalized_column]] = pd.DataFrame(minmax.transform(df_private_numerical[[normalized_column]].copy()))
    return df_train_numerical, df_public_numerical, df_private_numerical

def label_encoding(df_train, df_public, df_private, categorical_columns):
    print('label encoding')
    labelencoder = LabelEncoder()
    categorical_columns.remove("country")
    for categorical_column in categorical_columns:
        df_train[[categorical_column]] = pd.DataFrame(labelencoder.fit_transform(df_train[categorical_column].copy()))
        df_public[[categorical_column]] = pd.DataFrame(labelencoder.transform(df_public[categorical_column].copy()))
        df_private[[categorical_column]] = pd.DataFrame(labelencoder.transform(df_private[categorical_column].copy()))
    return df_train, df_public, df_private

def onehot_encoding(df_train, df_public, categorical_columns):
    print('One Hot encoding')

    print(list(df_train.columns))
    print(list(df_train.values[2]))
    
    onehot_encoder = OneHotEncoder()
    for categorical_column in categorical_columns:
        ohe_df = pd.DataFrame(onehot_encoder.fit_transform(df_train[[categorical_column]].copy()))
        df_train = pd.concat([df_train, ohe_df], axis=1).drop([categorical_column], axis=1)
        ohe_df = pd.DataFrame(onehot_encoder.transform(df_public[[categorical_column]].copy()))
        df_public = pd.concat([df_public, ohe_df], axis=1).drop([categorical_column], axis=1)
        print(list(df_train.columns))
        print(list(df_train.values[2]))
    # df_train = pd.get_dummies(df_train, columns = categorical_column)
    # df_public = pd.get_dummies(df_public, columns = categorical_column)
    return df_train, df_public

def get_preprocessed_data(args, load=True):
    if os.path.exists(args.train_preprocessed_pickle) and os.path.exists(args.public_preprocessed_pickle) and os.path.exists(args.private_preprocessed_pickle)and load:
        df_train = load_from_pickle(args.train_preprocessed_pickle)
        df_public = load_from_pickle(args.public_preprocessed_pickle)
        df_private = load_from_pickle(args.private_preprocessed_pickle)
    else:
        df_train = get_train_data(args)
        if args.origin_label:
            df_train['label'] = df_train['sar_flag']
        df_train = df_train.drop(columns=['cust_id', 'date'])
        
        df_public = get_public_data(args)
        if args.origin_label:
            df_public['label'] = df_public['sar_flag']
        df_public = df_public.drop(columns=['cust_id', 'date'])

        df_private = get_private_data(args)
        df_private['sar_flag'] = df_private['label']
        df_private = df_private.drop(columns=['cust_id', 'date'])

        df_train['alert_key'] = df_train['alert_key'].astype(int)
        df_public['alert_key'] = df_public['alert_key'].astype(int)
        df_private['alert_key'] = df_private['alert_key'].astype(int)

        df_train, df_public, df_private = missing_process(df_train, df_public, df_private)

        numerical_columns, categorical_columns = get_column_type(df_train)
        df_train[numerical_columns], df_public[numerical_columns], df_private[numerical_columns] = normalize(df_train, df_public, df_private, numerical_columns)

        df_train, df_public, df_private = label_encoding(df_train, df_public, df_private, categorical_columns)

        write_to_pickle(args.train_preprocessed_pickle, df_train)
        write_to_pickle(args.public_preprocessed_pickle, df_public)
        write_to_pickle(args.private_preprocessed_pickle, df_private)
    return df_train, df_public, df_private

def pred_to_csv(args, df_pred):
    # 考慮private alert key部分，滿足上傳條件

    public_private_test_csv = os.path.join(args.data_dir, '預測的案件名單及提交檔案範例.csv')
    df_public_private_test = pd.read_csv(public_private_test_csv)

    df_pred['alert_key'] = df_pred['alert_key'].astype(int)
    public_private_alert_key = df_public_private_test['alert_key'].values
    non_exists_keys = list(set(public_private_alert_key) - set(df_pred['alert_key'].unique()))
    non_exists_keys_index = pd.RangeIndex(len(non_exists_keys))
    df_non_exists_keys = pd.DataFrame(0, index=non_exists_keys_index, columns=df_pred.columns)
    df_non_exists_keys['alert_key'] = non_exists_keys
    df_predicted = pd.concat([df_pred, df_non_exists_keys])
    df_predicted.to_csv(args.pred_path, index=False)
    print('Output to csv')

def evaluate(args):
    df_pred = pd.read_csv(args.pred_path)
    df_ans = pd.read_csv(args.ans_path)
    df = df_pred.merge(df_ans)
    sar_id = np.where(df['sar_flag'] == 1)[0]
    n_sar = len(sar_id)
    print(f'score: {n_sar / sar_id[-1]}\t{n_sar} / {sar_id[-1]}')

def write_to_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_from_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_checkpoint(checkpoint_path, model):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    state = model.state_dict()
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    print('model loaded from %s' % checkpoint_path)
    return model