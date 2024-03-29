

# !pip install xgboost==1.7.1

# import library
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
import utils_base
import xgboost as xgb
from model import DNN_Model, DNN_Model_Prob
import torch
from dataset_base import label_Dataset, alert_key_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import copy
import torchvision
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import svm

def ML_model_train(Model, training_data, labels):
    # 使用訓練資料訓練模型
    Model.fit(training_data, labels)
    return Model

def ML_model_pred(Model, public_testing_data, public_testing_alert_key):
    predicted = []
    for i, _x in enumerate(Model.predict_proba(public_testing_data)):
        predicted.append([public_testing_alert_key[i], _x[1]])
    predicted = sorted(predicted, reverse=True, key= lambda s: s[0])
    return predicted

def ML_model_prob(Model, public_testing_data, public_testing_alert_key):
    predicted = ML_model_pred(Model, public_testing_data, public_testing_alert_key)
    prob = np.array(predicted)[:, 1]
    return prob

def seed_everything(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def run_one_epoch(
    args,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    mode='train'
):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    epoch_loss = 0.0
    epoch_correct = 0.0
    criterion = torch.nn.BCELoss()
    n_batch = len(dataloader)
    batch_pbar = tqdm((dataloader), total=n_batch, desc="Batch")
    with torch.set_grad_enabled(mode == "train"):
        for batch_idx, data_label in enumerate(batch_pbar, 1):
            data, label = data_label
            data = data.type(torch.FloatTensor).to(args.device)
            label = label.type(torch.FloatTensor).to(args.device)
            if mode == "train":
                optimizer.zero_grad()
            output = model(data)
            loss = torchvision.ops.sigmoid_focal_loss(output.squeeze(), label, alpha=0.95, reduction='mean')
            # loss = criterion(output.squeeze(), label)
            if mode == "train":
                loss.backward()
                optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss
            pred = torch.round(output).squeeze(-1)
            batch_correct = pred.eq(label.view_as(pred)).sum().item()
            epoch_correct += batch_correct
            batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
            batch_pbar.set_postfix(loss=f"{batch_loss:.4f}")

    if mode != "train":
        scheduler.step()

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / n_data
    performance["acc"] = epoch_correct / n_data
    return performance

def model_train(args, training_data, labels, testing_data, load=True):
    seed_everything(args)
    
    trainset = label_Dataset(training_data, labels)
    testing_label = pd.read_csv(args.ans_path)['sar_flag']
    valset = label_Dataset(testing_data, testing_label)
    # Use the torch dataloader to iterate through the dataset
    train_loader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = DNN_Model(n_feature=np.shape(training_data)[-1]).to(args.device)
    if os.path.exists(args.load) and load:
        model = utils_base.load_checkpoint(args.load, model)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_loss = np.inf
    best_acc = -np.inf
    best_model_weight = copy.deepcopy(model.state_dict())
    trigger_times = 0
    epoch_pbar = trange(args.epoch, desc="Epoch")
    # For each epoch
    for epoch_idx in epoch_pbar:
        performance_train = run_one_epoch(args, model, train_loader, optimizer, scheduler, 'train')
        performance_eval = run_one_epoch(args, model, val_loader, optimizer, scheduler, 'eval')

        if epoch_idx % args.save_interval == 0:
            utils_base.save_checkpoint(os.path.join(args.save, f"{epoch_idx+1}.pth"), model)
            
        if args.matrix == "loss":
            if performance_eval["loss"] < best_loss:
                best_loss = performance_eval["loss"]
                best_model_weight = copy.deepcopy(model.state_dict())
                trigger_times = 0
                utils_base.save_checkpoint(os.path.join(args.save, "better.pth"), model)
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    model.load_state_dict(best_model_weight)
                    break
        else:
            if performance_eval > best_acc:
                best_acc = performance_eval
                best_model_weight = copy.deepcopy(model.state_dict())
                trigger_times = 0
                utils_base.save_checkpoint(os.path.join(args.save, "better.pth"), model)
            else:
                trigger_times += 1
                if trigger_times >= args.epoch_patience:
                    print("Early Stop")
                    model.load_state_dict(best_model_weight)
                    break

        epoch_pbar.set_description(f"Epoch [{epoch_idx+1}/{args.epoch}]")
        epoch_pbar.set_postfix(
            train_loss=performance_train["loss"],
            train_acc=performance_train["acc"],
            eval_loss=performance_eval["loss"],
            eval_acc=performance_eval["acc"],
        )
    model.load_state_dict(best_model_weight)
    utils_base.save_checkpoint(os.path.join(args.save, "best.pth"), model)
    return model

def model_pred(
    args,
    model: torch.nn.Module,
    testing_data,
    testing_alert_key
):
    testset = alert_key_Dataset(testing_data, testing_alert_key)
    # Use the torch dataloader to iterate through the dataset
    test_loader = DataLoader(
        testset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )

    model.eval()
    prediction = []
    n_batch = len(test_loader)
    batch_pbar = tqdm((test_loader), total=n_batch, desc="Batch")
    for batch_idx, data_alert_key in enumerate(batch_pbar, 1):
        data, alert_key = data_alert_key
        data = data.type(torch.FloatTensor).to(args.device)
        output = model(data)
        prob = output.squeeze(-1).detach().cpu().numpy()
        alert_key = alert_key.squeeze(-1).detach().cpu().numpy()
        key_prob = np.dstack((alert_key, prob)).squeeze(0)
        prediction.extend(key_prob)
        batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
    
    prediction = sorted(prediction, reverse=True, key= lambda s: s[0])
    return prediction

def pred_to_csv(args, df_pred, df_public_private_test):
    df_pred['alert_key'] = df_pred['alert_key'].astype(int)
    df_pred = df_pred.groupby('alert_key').mean().reset_index()
    alert_keys = df_pred['alert_key']
    predicted = df_pred.to_numpy().tolist()
    # 考慮private alert key部分，滿足上傳條件
    public_private_alert_key = df_public_private_test['alert_key'].values
    # print(len(public_private_alert_key))

    # For alert key not in public, add zeros
    for key in public_private_alert_key:
        if key not in alert_keys:
            predicted.append([key, 0])

    predict_alert_key, predict_probability = [], []
    for key, prob in predicted:
        predict_alert_key.append(key)
        predict_probability.append(prob)

    df_predicted = pd.DataFrame({
        "alert_key": predict_alert_key,
        "probability": predict_probability
    })
    df_predicted = df_predicted.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
    df_predicted['alert_key'] = df_predicted['alert_key'].astype(int)
    df_predicted = df_predicted.groupby('alert_key').mean().reset_index()
    df_predicted['alert_key'] = df_predicted['alert_key'].astype(int)
    df_predicted.to_csv(args.pred_path, index=False)
    print('output csv to', args.pred_path)

def evaluate(args):
    df_pred = pd.read_csv(args.pred_path)
    df_ans = pd.read_csv(args.ans_path)
    df = df_pred.merge(df_ans)
    sar_id = np.where(df['sar_flag'] == 1)[0]
    n_sar = len(sar_id)
    print(f'score: {n_sar / sar_id[-1]}\t{n_sar} / {sar_id[-1]}')

def main(args):
    """# 訓練資料處理"""
    data_dir = './data'
    # data preprocessing
    training_data, labels, public_testing_data, public_testing_alert_key, private_testing_data, private_testing_alert_key = utils_base.get_data(data_dir=args.data_dir, preprocess_data_dir=args.preprocess_data_dir)
    # print('first training data', training_data[0])
    # print(training_data.shape, public_testing_data.shape)
    print('training_data', np.shape(training_data))
    print('labels', np.shape(labels))
    print('public_testing_data', np.shape(public_testing_data))
    print('public_testing_alert_key', np.shape(public_testing_alert_key))
    print('private_testing_data', np.shape(private_testing_data))
    print('private_testing_alert_key', np.shape(private_testing_alert_key))

    """## 缺失值補漏
    可以發現有不少筆資料其實是有缺漏的，補上缺失值的方法有很多種，我們對於數值類資料補上中位數，對於類別類資料補上眾數。
    """

    ''' Missing Value Imputation '''
    # for numerical index we do imputation using median
    numerical_index = [2,4,5,6,7,8,9,10,11,14,17,24]
    # Otherwise we select the most frequent
    non_numerical_index = [0,1,3,12,13,15,16,18,19,20,21,22,23]


    training_data, public_testing_data, private_testing_data, numerical_index, non_numerical_index, labels = utils_base.missing_imputate(training_data, public_testing_data, private_testing_data, numerical_index, non_numerical_index, labels)

    """ normalization"""
    # training_data, public_testing_data = utils_base.normalize(training_data, public_testing_data, numerical_index, non_numerical_index)

    """  此外，若類別類資料跟數字大小沒關係，我們採用 one-hot encoding 將其編碼。"""

    # for some catogorical features, we do one hot encoding
    # one_hot_index = [1,3,4,5,6,7,8,9,12]

    one_hot_index = non_numerical_index
    # one_hot_index = [1,5,6,7,10,11]

    training_data, public_testing_data, private_testing_data = utils_base.onehot_encoding(training_data, public_testing_data, private_testing_data, one_hot_index)
    training_data = training_data.toarray()
    public_testing_data = public_testing_data.toarray()
    private_testing_data = private_testing_data.toarray()
    # """# XGBoost 訓練"""
    # # 建立 XGBClassifier 模型
    xgbrModel = xgb.XGBClassifier(
        n_estimators=195,         # 樹的個數--1000棵樹建立xgboost
        random_state=0            # 隨機數
    )

    # # 定義參數各種組合
    # n_estimators = [x * 100 for x in range(8, 13)]
    # learn_rate = [x * 0.1 for x in range(1, 5)]
    # gamma = [x * 0.1 for x in range(3)]
    # param_grid = dict(n_estimators=n_estimators, learning_rate=learn_rate, gamma=gamma)
    # grid = GridSearchCV(estimator=xgbrModel, param_grid=param_grid)
    # grid.fit(training_data, labels)
    xgbrModel = ML_model_train(xgbrModel, training_data, labels)
    # xgbrModel = grid

    # """# RandomForestClassifier 訓練"""
    # RFC = RandomForestClassifier(n_estimators = 100)
    # RFC = ML_model_train(RFC, training_data, labels)

    # """# KNeighborsClassifier 訓練"""
    # KNN = KNeighborsClassifier(n_neighbors=3)
    # KNN = ML_model_train(KNN, training_data, labels)

    # """# DecisionTreeClassifier 訓練"""
    # DT = DecisionTreeClassifier()
    # DT = ML_model_train(DT, training_data, labels)
    
    """# SVM 訓練"""
    SVM = svm.SVC(kernel='poly', degree=3, C=1.0, probability=True)
    SVM = ML_model_train(SVM, training_data, labels)


    """# Resnet 訓練"""
    dnn = model_train(args, training_data, labels, public_testing_data, load=False)

    """# 預測與結果輸出
    利用訓練好的模型對目標alert key預測報SAR的機率以及輸出為目標格式。
    目標輸出筆數3850，其中public筆數為1845筆。
    因上傳格式需要private跟public alert key皆考慮，直接從預測範本統計要預測的alert key，預測結果輸出為prediction.csv。
    """
    
    # Read csv of all alert keys need to be predicted
    public_private_test_csv = os.path.join(data_dir, '預測的案件名單及提交檔案範例.csv')
    df_public_private_test = pd.read_csv(public_private_test_csv)

    # # Predict probability
    # predicted_xgbr = ML_model_pred(xgbrModel, public_testing_data, public_testing_alert_key)
    prob_xgbr_public = ML_model_prob(xgbrModel, public_testing_data, public_testing_alert_key)

    # prob_RFC_public = ML_model_prob(RFC, public_testing_data, public_testing_alert_key)

    # prob_KNN_public = ML_model_prob(KNN, public_testing_data, public_testing_alert_key)

    # prob_DT_public= ML_model_prob(DT, public_testing_data, public_testing_alert_key)

    prob_SVM_public = ML_model_prob(SVM, public_testing_data, public_testing_alert_key)

    dnn_prob = DNN_Model_Prob(n_feature=np.shape(training_data)[-1]).to(args.device)
    dnn_prob.load_state_dict(dnn.state_dict())
    predicted_DNN = model_pred(args, dnn_prob, public_testing_data, public_testing_alert_key)
    prob_DNN_public = np.array(predicted_DNN)[:, 1]
    
    df_pred_public = pd.DataFrame(predicted_DNN, columns = ['alert_key', 'probability'])
    # df_pred['probability'] = prob_xgbr * 5 + prob_RFC + prob_KNN + prob_DT + prob_SVM + prob_DNN * 3
    df_pred_public['probability'] = prob_DNN_public + prob_xgbr_public * 5 + prob_SVM_public
    # df_pred['probability'] = prob_DNN

    pred_to_csv(args, df_pred_public, df_public_private_test)
    evaluate(args)

    # predicted_xgbr = ML_model_pred(xgbrModel, private_testing_data, private_testing_alert_key)
    prob_xgbr_private = ML_model_prob(xgbrModel, private_testing_data, private_testing_alert_key)

    # prob_RFC_private = ML_model_prob(RFC, private_testing_data, private_testing_alert_key)

    # prob_KNN_private = ML_model_prob(KNN, private_testing_data, private_testing_alert_key)

    # prob_DT_private = ML_model_prob(DT, private_testing_data, private_testing_alert_key)

    prob_SVM_private = ML_model_prob(SVM, private_testing_data, private_testing_alert_key)

    predicted_DNN = model_pred(args, dnn_prob, private_testing_data, private_testing_alert_key)
    prob_DNN_private = np.array(predicted_DNN)[:, 1]

    df_pred_private = pd.DataFrame(predicted_DNN, columns = ['alert_key', 'probability'])
    # df_pred['probability'] = prob_xgbr * 5 + prob_RFC + prob_KNN + prob_DT + prob_SVM + prob_DNN * 3
    df_pred_private['probability'] = prob_DNN_private + prob_xgbr_private * 5 + prob_SVM_private
    # df_pred['probability'] = prob_DNN

    df_pred = pd.concat([df_pred_public, df_pred_private])
    df_pred.sort_values(by=['probability'], ascending=False)

    pred_to_csv(args, df_pred, df_public_private_test)
    
def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--seed", default=100, type=int, help="the seed (default 100)")
    parser.add_argument("--save", default='./checkpoints_base', type=str, help="the directory to save checkpoints.")
    parser.add_argument("--load", default='./checkpoints_base/best.pth', type=str, help="the directory to load the checkpoint.")
    parser.add_argument("--data_dir", default='./data', type=str, help="the directory to csv files.")
    parser.add_argument("--pred_path", default='./prediction_baseline_0.csv', type=str, help="the path to pred file.")
    parser.add_argument("--ans_path", default='./data/24_ESun_public_y_answer.csv', type=str, help="the path to ans file.")
    parser.add_argument("--preprocess_data_dir", default='./preprocess_data', type=str, help="the directory to preprocessed pickle files.")
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--workers", default=8, type=int, help="the number of data loading workers (default: 4)")

    # optimizer
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # data loader
    parser.add_argument("--train_batch", type=int, default=128)
    parser.add_argument("--test_batch", type=int, default=128)

    # training
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--epoch_patience", type=int, default=20)

    parser.add_argument("--matrix", type=str, default='loss')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # loadData()
    main(args)