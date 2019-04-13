import os
import csv
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import time
import matplotlib.cm as cm
from utils.constants import IMAGES_DATASETS
# import cv2
sns.set(color_codes=True)


def get_percentile(scores, dataset):
    if dataset == 'kdd':
        # Highest 20% are anomalous
        per = np.percentile(scores, 80)
    elif dataset == "arrhythmia":
        # Highest 15% are anomalous
        per = np.percentile(scores, 85)
    else:
        c = 90
        per = np.percentile(scores, 100 - c)
    return per



def predict(scores, threshold):
    return scores>=threshold


def save_results(scores, data, model, dataset, method, weight, label,
                 random_seed, step=-1,train=False):

    directory = 'results/{}/{}/{}/w{}/'.format(model,
                                                  dataset,
                                                  method,
                                                  weight)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = "{}_step{}_rd{}".format(dataset, step, random_seed)
    fname = directory + "results.csv"

    
    scores = np.array(scores) 
    per = 0
    if method == 'ch':
        per = 1371.97
    if method == 'l1':
        per = 141552.62
    if method == 'l2':
        per = 55937.08
    if method == 'fm':
        per = 259585.21
    y_pred = (scores<=per)

    #results = [model, dataset, method, weight, label,
      #         step, roc_auc, precision, recall, f1, random_seed, time.ctime()]
    if method == 'fm':
        save_results_csv("results/results.csv", data, y_pred)
    
    #results = [step, roc_auc, precision, recall, f1, random_seed]
    #save_results_csv(fname, results)



def save_results_csv(fname, data, y_pred):
    """Saves results in csv file
    Args:
        fname (str): name of the file
        results (list): list of prec, rec, F1, rds
    """

    
    if not os.path.isfile(fname):
        args = fname.split('/')[:-1]
        directory = os.path.join(*args)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(fname, 'wt') as f:
            writer = csv.writer(f)
            writer.writerows(
                    [['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
        'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 
        'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 
        'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 
        'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 
        'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']])

    new_rows = []
    with open(fname, 'at') as f:
        # Overwrite the old file with the modified rows
        for x in range(len(data)):
            writer = csv.writer(f)
            for y in range(len(data[0])):
                new_rows.append(data[x][y])
            new_rows.append(0 if y_pred[x] == False else 1)
            writer.writerow(new_rows)
            new_rows = []
        
        