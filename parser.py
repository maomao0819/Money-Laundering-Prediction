from argparse import ArgumentParser, Namespace
import os
import torch

def boolean_string(str):
    return ("t" in str) or ("T" in str)

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--seed", default=100, type=int, help="the seed (default 100)")
    parser.add_argument("--save", default='./checkpoints', type=str, help="the directory to save checkpoints.")
    parser.add_argument("--load", default='./checkpoints/best.pth', type=str, help="the directory to load the checkpoint.")
    parser.add_argument("--data_dir", default='./data', type=str, help="the directory to csv files.")
    parser.add_argument("--pred_path", default='./prediction_baseline.csv', type=str, help="the path to pred file.")
    parser.add_argument("--ans_path", default='./data/24_ESun_public_y_answer.csv', type=str, help="the path to ans file.")
    parser.add_argument("--preprocess_data_dir", default='./preprocess_data', type=str, help="the directory to preprocessed pickle files.")
    parser.add_argument("--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda")
    parser.add_argument("--workers", default=8, type=int, help="the number of data loading workers (default: 4)")
    parser.add_argument("--n_day_range", default=5, type=int)
    parser.add_argument("--df_batch_size", default=128, type=int)
    parser.add_argument("--origin_label", default=True, type=bool)
    parser.add_argument("--train_pickle", default='./preprocess_data/train_origin_label_5.pickle', type=str)
    parser.add_argument("--public_pickle", default='./preprocess_data/public_origin_label_5.pickle', type=str)
    parser.add_argument("--private_pickle", default='./preprocess_data/private_origin_label_5.pickle', type=str)
    parser.add_argument("--train_preprocessed_pickle", default='./preprocess_data/train_preprocessed_origin_label_5.pickle', type=str)
    parser.add_argument("--public_preprocessed_pickle", default='./preprocess_data/public_preprocessed_origin_label_5.pickle', type=str)
    parser.add_argument("--private_preprocessed_pickle", default='./preprocess_data/private_preprocessed_origin_label_5.pickle', type=str)
    # optimizer
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)

    # data loader
    parser.add_argument("--train_batch", type=int, default=16384)
    parser.add_argument("--test_batch", type=int, default=16384)

    # training
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--epoch_patience", type=int, default=100)

    parser.add_argument("--matrix", type=str, default='loss')
    args = parser.parse_args()
    return args
