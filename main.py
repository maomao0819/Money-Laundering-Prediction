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
from engine import predict_all

def main(args):
    df_train, df_public, df_private = utils.get_preprocessed_data(args)
    predict_all(args, df_train, df_public, df_private)

if __name__ == "__main__":
    args = parse_args()
    main(args)
