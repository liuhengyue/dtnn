import json
import glob
import os
import numpy as np
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
# plt.tight_layout()
def load_results():
    """
    load results json file by locating the latest .json file in logs/ dir.
    :return: results dict
    """
    results_files = glob.glob('logs/*.json')
    print(results_files)
    latest_results_file_path = max(results_files, key=os.path.getctime)
    with open(latest_results_file_path, 'r') as f:
        latest_results_file = f.read()
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
    img_ids = list(results.keys())
    # print(img_ids)
    index = np.array(img_ids[:100])
    # index = pd.date_range("1 1 2000", periods=100, freq = "m", name = "date")
    data = np.random.randn(index.size, 4).cumsum(axis=0)
    # init plot
    fig = plt.figure()
    ax = plt.axes()

    # plot data
    ax.plot(index, data)
    ax.grid(linestyle='-', color='white')
    ax.legend(['1', '2', '3', '4'], loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, facecolor='black')
    # plot configurations
    start, end = ax.get_xlim()
    stepsize = 10
    ax.xaxis.set_ticks(np.arange(start, end, stepsize))

    # labels
    # plt.title("Controller")
    plt.xlabel("Image index", labelpad=100)
    plt.ylabel("Utilization parameter u")

    from matplotlib.font_manager import findfont, FontProperties
    font = findfont(FontProperties(family=['serif']))
    print(font)

    plt.show()

if __name__ == "__main__":
    results = load_results()
    u_vs_examples_plot(results)