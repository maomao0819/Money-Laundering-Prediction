import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import utils
import torch
from dataset import label_Dataset, alert_key_Dataset
import torchvision
from model import DNN_Model, DNN_Model_Prob
from torch.utils.data import DataLoader
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb

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
            loss = torchvision.ops.sigmoid_focal_loss(output.squeeze(), label, alpha=0.001, reduction='mean')
            if mode == "train":
                loss.backward()
                optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss  # sum up batch loss
            batch_pbar.set_description(f"Batch [{batch_idx}/{n_batch}]")
            batch_pbar.set_postfix(loss=f"{batch_loss:.4f}")

    if mode != "train":
        scheduler.step()

    performance = {}
    n_data = len(dataloader.dataset)
    performance["loss"] = epoch_loss / n_data
    return performance

def model_train(args, df_train, df_public, load=True):
    seed_everything(args)

    trainset = label_Dataset(df_train)
    valset = label_Dataset(df_public)

    # Use the torch dataloader to iterate through the dataset
    train_loader = DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    model = DNN_Model(n_feature=len(df_train.columns)-2).to(args.device)
    if os.path.exists(args.load) and load:
        model = utils.load_checkpoint(args.load, model)

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
            utils.save_checkpoint(os.path.join(args.save, f"{epoch_idx+1}.pth"), model)
            

        if performance_eval["loss"] < best_loss:
            best_loss = performance_eval["loss"]
            best_model_weight = copy.deepcopy(model.state_dict())
            trigger_times = 0
            utils.save_checkpoint(os.path.join(args.save, "better.pth"), model)
        else:
            trigger_times += 1
            if trigger_times >= args.epoch_patience:
                print("Early Stop")
                model.load_state_dict(best_model_weight)
                break

        epoch_pbar.set_description(f"Epoch [{epoch_idx+1}/{args.epoch}]")
        epoch_pbar.set_postfix(
            train_loss=performance_train["loss"],
            eval_loss=performance_eval["loss"],
        )
    model.load_state_dict(best_model_weight)
    utils.save_checkpoint(os.path.join(args.save, "best.pth"), model)
    return model

def model_pred(
    args,
    model: torch.nn.Module,
    df_test
):
    testset = alert_key_Dataset(df_test)
    # Use the torch dataloader to iterate through the dataset
    test_loader = DataLoader(
        testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True
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

def model_prob(args, df_train, df_public, df_private, pred_type='public', load=False, load_to_train=False):
    df_train = df_train.drop(columns=['sar_flag'])
    df_public = df_public.drop(columns=['sar_flag'])
    df_private = df_private.drop(columns=['sar_flag'])
    if os.path.exists(args.load) and load:
        model = DNN_Model(n_feature=len(df_train.columns)-2).to(args.device)
        dnn = utils.load_checkpoint(args.load, model)
    else:
        dnn = model_train(args, df_train, df_public, load=load_to_train)
    dnn_prob = DNN_Model_Prob(n_feature=len(df_train.columns)-2).to(args.device)
    dnn_prob.load_state_dict(dnn.state_dict())
    if pred_type == 'public':
        pred = model_pred(args, dnn_prob, df_public)
    else:
        pred = model_pred(args, dnn_prob, df_private)
    df_pred = pd.DataFrame(pred, columns = ['alert_key','probability'])
    df_pred = df_pred.groupby('alert_key').mean().reset_index()
    df_pred = df_pred.sort_values(by=['alert_key'], ascending=True).reset_index(drop=True)
    return df_pred

def ML_model_prob(model, df_train, df_test, label_column='sar_flag'):
    model = make_pipeline(model)
    df_train_data = df_train.drop(columns=['sar_flag', 'alert_key', 'label'])
    df_test_data = df_test.drop(columns=['sar_flag', 'alert_key', 'label'])
    model.fit(df_train_data, df_train[label_column])
    predicted_probs = []
    for predicted_prob in model.predict_proba(df_test_data):
        predicted_probs.append(predicted_prob[1])

    pred_data = {'alert_key': df_test['alert_key'],
            'probability': predicted_probs
    }
    df_pred = pd.DataFrame(pred_data)
    df_pred = df_pred.groupby('alert_key').mean().reset_index()
    df_pred = df_pred.sort_values(by=['alert_key'], ascending=True).reset_index(drop=True)
    print('Predict Finish')
    return df_pred

def predict(args, df_train, df_public, df_private, pred_type='public'):
    if pred_type == 'public':
        df_test = df_public
    else:
        df_test = df_private

    xgbrModel = xgb.XGBClassifier(learning_rate=0.1,
        n_estimators=1000,         # 樹的個數--1000棵樹建立xgboost
        max_depth=6,               # 樹的深度
        min_child_weight = 1,      # 葉子節點最小權重
        gamma=0.,                  # 懲罰項中葉子結點個數前的參數
        subsample=0.8,             # 隨機選擇80%樣本建立決策樹
    #   objective='multi:softmax', # 指定損失函數
        scale_pos_weight=1,        # 解決樣本個數不平衡的問題
        random_state=0            # 隨機數
    )

    """# RandomForestClassifier 訓練"""
    RFC = RandomForestClassifier(n_estimators = 100)

    """# KNeighborsClassifier 訓練"""
    KNN = KNeighborsClassifier(n_neighbors=3)

    """# DecisionTreeClassifier 訓練"""
    DT = DecisionTreeClassifier()
    
    """# SVM 訓練"""
    svc = svm.SVC(C=1.0, probability=True)
    svr = svm.SVR(C=1.0, epsilon=0.2)

    df_pred_xgbr = ML_model_prob(xgbrModel, df_train, df_test, label_column='sar_flag')
    # df_pred_RFC = ML_model_prob(RFC, df_train, df_test, label_column='sar_flag')
    # df_pred_KNN = ML_model_prob(KNN, df_train, df_test, label_column='sar_flag')
    # df_pred_DT = ML_model_prob(DT, df_train, df_test, label_column='sar_flag')
    # df_pred_SVC = ML_model_prob(svc, df_train, df_test, label_column='sar_flag')
    # df_pred_SVR = ML_model_prob(svr, df_train, df_test, label_column='label')
    df_pred_dnn = model_prob(args, df_train, df_public, df_private, pred_type=pred_type, load=args.load_model, load_to_train=args.load_to_train)

    # pred_prob = 10 * df_pred_xgbr['probability'] + df_pred_RFC['probability'] + df_pred_KNN['probability'] + \
    #     df_pred_DT['probability'] + df_pred_SVC['probability'] + 2 * df_pred_SVR['probability'] + 12 * df_pred_dnn['probability']
    pred_prob = df_pred_xgbr['probability'] * 5 + df_pred_dnn['probability']
    pred_data = {'alert_key': df_pred_dnn['alert_key'],
            'probability': pred_prob
    }
    df_pred = pd.DataFrame(pred_data)
    df_pred = df_pred.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
    df_pred['alert_key'] = df_pred['alert_key'].astype(int)
    utils.pred_to_csv(args, df_pred)
    if pred_type == 'public':
        utils.evaluate(args)
    return df_pred

def predict_all(args, df_train, df_public, df_private):

    df_pred_public = predict(args, df_train, df_public, df_private, pred_type='public')
    df_pred_private = predict(args, df_train, df_public, df_private, pred_type='private')
    df_pred = pd.concat([df_pred_public, df_pred_private])
    df_pred = df_pred.sort_values(by=['probability'], ascending=False).reset_index(drop=True)
    utils.pred_to_csv(args, df_pred)