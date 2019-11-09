import json
import glob
import os
import numpy as np
from collections import defaultdict
# import pandas as pd
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
# sns.set()
# plt.close("all")
# matplotlib params
# print(plt.style.available)
# plt.style.use('fivethirtyeight')
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = 15, 10
rcParams['font.size'] = 25
rcParams['axes.facecolor'] = '#ebebeb'
rcParams['figure.autolayout'] = True
plt.style.use('ggplot')
# plt.tight_layout()
def load_logs():
    """
    load results json file by locating the latest .json file in logs/ dir.
    :return: results dict
    """
    results_files = glob.glob('logs/*.json')
    latest_results_file_path = max(results_files, key=os.path.getctime)
    with open(latest_results_file_path, 'r') as f:
        latest_results_file = f.read()
        print("load from {}".format(latest_results_file_path))
    # results format:
    # {
    #   0 : {'label' : 1 , 'prediction' : { 0.1 : 0, 0.2 : 1, ...} },
    #   1 : {...},
    #   ...
    # }
    # to access:
    #   results[img_id]['label'] gives the label
    #   results[img_id]['prediction'][0.1] gives the prediction when u == 0.1
    results = json.loads(latest_results_file)
    return results


def get_best_tradeoff_u(result):
    """

    :param result:
    :return: [u_0, u_1, ..., u_best]
    """
    preds = []
    label = result["label"]
    u = 2.0
    for k, v in result["prediction"].items():
        # also get pred for each u
        if v == label:
            preds.append(float(k))
        else:
            preds.append(0)
        if v == label and float(k) < u:
            u = float(k)
    if u == 2.0:
        u = 0
    preds.append(u)
    return preds


def process_logs(logs):
    """
    Define a format here:
    {
        'ids' : [image_id_0, ...]
      if correct: u=0.1, u=0.2, ..., best_u
         'us' : [[0.1, 0.7,...],...]
        ...
    }
    :return:
    """
    results = defaultdict(list)
    for k, v in logs.items():
        results["ids"].append(k)
        results["us"].append(get_best_tradeoff_u(v))
    return results


def u_vs_examples_plot(results):
    """
    for this visualization, the x-axis is each example,
    and y-axis is the u value. We draw a point if it achieves
    right prediction with the lowest u.
    We also draw 3 similar lines for constant u = 0.1, 0.5, 1.0.
    Then we showcase our context-aware control achieves the
    best best best trade-off.
    :param results: results dict
    :return: None
    """
    # declarations of variables
    u_list = []
    img_ids = results["ids"]
    # print(img_ids)
    index = np.array(img_ids)
    data = np.array(results["us"])
    best_u = np.sort(data[:, -1])

    # test
    # index = index[:1000]
    # data = data[:1000]

    # data, index = zip(*sorted(zip(best_u, index)))

    # init plot
    fig = plt.figure()
    ax = plt.axes()
    ax.set_axisbelow(True)
    # plot data
    ax.grid(linestyle='-', color='white')

    # best u case
    ax.plot(index, best_u)
    plt.fill_betweenx(index, best_u, alpha=0.5)

    # u = 0.5
    u_5 = np.sort(data[:, 4])
    ax.plot(index, u_5)
    plt.fill_betweenx(index, u_5, alpha=0.5)

    # u = 1.0
    u_10 = np.sort(data[:, 9])
    ax.plot(index, u_10)
    plt.fill_betweenx(index, u_10, alpha=0.5)

    ax.legend(['1', '2', '3'], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, facecolor='black')
    # plot configurations
    # start, end = ax.get_xlim()
    start, end = 0, index.size
    stepsize = 2000
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))
    # ax.xaxis.set_ticks([])
    # labels
    # plt.title("Controller")
    plt.xlabel("Image index (sorted by u)", labelpad=100)
    plt.ylabel("Utilization parameter u")


    plt.show()

if __name__ == "__main__":
    logs = load_logs()
    results = process_logs(logs)
    # print(results)
    u_vs_examples_plot(results)