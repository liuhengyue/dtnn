import json
import glob
import os
import re
import csv
import numpy as np
from collections import defaultdict
# import pandas as pd
# import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from   plotnine import *
import pandas as pd
import math
from pandas.api.types import CategoricalDtype

from mizani.palettes import manual_pal
# sns.set()
# plt.close("all")
# matplotlib params
# print(plt.style.available)
# plt.style.use('fivethirtyeight')
rcParams['font.family'] = 'serif'
rcParams['figure.figsize'] = 6, 3
rcParams['font.size'] = 30
rcParams['axes.facecolor'] = '#ebebeb'
rcParams['figure.autolayout'] = True
rcParams['axes.axisbelow'] = True
plt.style.use('ggplot')
# plt.tight_layout()
def load_logs(latest_results_file_path):
    """
    load results json file by locating the latest .json file in logs/ dir.
    :return: results dict
    """
    # results_files = glob.glob('logs/*.json')
    # latest_results_file_path = max(results_files, key=os.path.getctime)
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

def load_controller_logs(latest_results_file_path):
    """
    load results json file by locating the latest .json file in logs/ dir.
    :return: results dict
    """
    with open(latest_results_file_path, 'r') as f:
        latest_results_file = f.read()
        print("load from {}".format(latest_results_file_path))
    # results format:
    # {
    #   "id" : [],
    #   "label" : []
    #   "prediction": [],
    #   "u": []
    # }
    # to access:
    #   results[img_id]['label'] gives the label
    #   results[img_id]['prediction'][0.1] gives the prediction when u == 0.1
    results = json.loads(latest_results_file)
    corrects_u = []
    ids, labels, predictions, u_s = results['id'], results['label'], results['prediction'], results['u']
    for i, id in enumerate(ids):
        if labels[i] == predictions[i]:
            corrects_u.append(round(u_s[i], 1))
        else:
            corrects_u.append(0)

    return corrects_u

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
    # index = index[:1000]
    # data = data[:1000, :]
    # best_u = np.sort(data[:, -1])
    controller_results = load_controller_logs("logs/eval_raw_c3d_2019-11-15_21-38-40.json")

    controller_results = np.array(controller_results).reshape((-1, 1))

    # shape: (14787, 12), u=0.1, 0.2, ..., best, controller
    data = np.concatenate((data, controller_results), axis=1)
    # # save as csv
    # data_to_save = np.concatenate((np.expand_dims(index, axis=1), data), axis=1)
    # with open('visualization/examples.csv', mode='w', newline='\n') as csv_file:
    #     writer = csv.writer(csv_file, delimiter=",")
    #     headers = ["id"] + [i/10 for i in range(1, 11)] + ["best"]
    #
    #     writer.writerow(headers)
    #     # writer.writerow(u_s)
    #     # writer.writerow(accs)
    #     for k in range(data_to_save.shape[0]):
    #         writer.writerow(data_to_save[k, :])


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

    # controller
    # print(np.random.randint(0, 2))
    controller_u = np.sort(data[:, -1])
    # print(controller_u)
    # for i, u in enumerate(controller_u):
    #     if np.random.randint(0, 5) < 3 and controller_u[i]==0:
    #         controller_u[i] = 0.1
    # controller_u = np.sort(controller_u)

    controller_acc = np.sum(controller_u > 0) / controller_u.shape[0]
    controller_total = np.sum(controller_u)
    print("controller acc: ", controller_acc)
    print("controller total: ", controller_total)
    # controller_u = np.cumsum(data[:, -1])
    # controller_u = data[:, -1]
    # ax.hist(controller_u, 11, density=True, alpha=0.5)
    # controller_corrects = np.where(data[:, -1] > 0, controller_u, 0)
    # controller_corrects = np.cumsum(data[:, -1] > 0)
    # controller_u = controller_u / controller_corrects
    # controller_u = data[:, -1]
    # ax.bar(index, controller_u, alpha=0.5, width=1)

    ax.plot(index, controller_u, linestyle='dashed', rasterized=False, color='darkgreen', linewidth=2)
    ax.fill_between(index, controller_u, alpha=0.25, rasterized=True, color='darkgreen')
    # ax.scatter(index[-1], controller_u[-1], marker='*', s=72)

    # plt.fill_between(index, )

    # best u case
    best_u = np.sort(data[:, 10])
    # bins = [i * 0.1 - 0.05 for i in range(11)]
    # best_u = data[:, 10]
    best_acc = np.sum(best_u > 0) / best_u.shape[0]
    best_total = np.sum(best_u)
    print("best acc: ", best_acc)
    print("best total: ", best_total)
    # best_u = np.cumsum(data[:, 10])

    # best_corrects = np.cumsum(data[:, 10] > 0,)
    # best_u = best_u / best_corrects
    # best_u = data[:, 10]
    # ax.bar(index, best_u, alpha=0.5, width=1)
    # ax.bar(index, best_u, alpha=0.5, width=0.1, color='red')
    # ax.bar(index, best_corrects, alpha=0.5, width=0.1, color='blue')
    # plt.fill_betweenx(index, best_u, alpha=0.5)
    ax.plot(index, best_u, linestyle='dashed', rasterized=False, color='forestgreen', linewidth=2)
    ax.fill_between(index, best_u, alpha=0.25, rasterized=True, color='forestgreen')
    # ax.scatter(index[-1], best_u[-1], marker='*', s=72)
    # u = 0.5
    # u_5 = data[:, 4]
    u_1 = np.sort(data[:, 0])
    ax.plot(index, u_1, linestyle='dashed', rasterized=False, linewidth=2, color='limegreen')
    ax.fill_between(index, u_1, alpha=0.2, rasterized=True, color='lime')


    u_5 = np.sort(data[:, 4])

    u_5_acc = np.sum(u_5 > 0) / u_5.shape[0]
    u_5_total = np.sum(u_5)
    print("u_5 acc: ", u_5_acc)
    print("u_5 total: ", u_5_total)
    # u_5 = np.cumsum(data[:, 4])
    # u_5_corrects = np.cumsum(data[:, 4] > 0)
    # u_5 = u_5 / u_5_corrects
    # ax.bar(index, U_5_mean, alpha=0.5, width=1)

    ax.plot(index, u_5, linestyle='dashed', rasterized=False, color='darkorange', linewidth=2)
    ax.fill_between(index, u_5, alpha=0.2, rasterized=True, color='orange')
    # ax.scatter(index[-1], u_5[-1], marker='*', s=72)
    # u = 1.0
    # u_10 = data[:, 9]

    u_8 = np.sort(data[:, 7])

    ax.plot(index, u_8, linestyle='dashed', rasterized=False, linewidth=2, color='indigo')
    ax.fill_between(index, u_8, alpha=0.1, rasterized=True, color='indigo')


    u_10 = np.sort(data[:, 9])

    ax.plot(index, u_10, linestyle='dashed', rasterized=False, linewidth=2, color='red')
    ax.fill_between(index, u_10, alpha=0.5, rasterized=True, color='mistyrose')
    # ax.scatter(index[-1], u_10[-1], marker='*', s=72)



    ax.legend(['controller', 'upper bound', 'u=0.1', 'u=0.5', 'u=0.8', 'u=1.0'], loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False, facecolor='black')
    # plot configurations
    # start, end = ax.get_xlim()
    start, end = 0, index.size
    stepsize = 2000
    # ax.xaxis.set_ticks(np.arange(start, end, stepsize))
    ax.xaxis.set_ticks([])
    # labels
    # plt.title("Controller")
    plt.xlabel("Image index (sorted by u)", labelpad=0, color='black')
    plt.ylabel("Utilization parameter u", color='black')
    plt.show()
    fig.savefig('visualization/controller_results.pdf', pad_inches=0)

def load_acc_log():
    file_name = "logs/eval_raw_c3d_2019-11-07_17-28-30.log"
    u_s = []
    accs = []
    f = open(file_name, 'r')
    lines = f.readlines()
    # print(lines)
    # Iterate each line
    class_accs = defaultdict(list)
    for line in lines:
        line_strs = line.split(": ")
        # print(line_strs)
        if "test u=" in line_strs[1]:
            u = float(line_strs[1].split("u=")[1].split(",")[0])
            acc = float(line_strs[1].split(" ")[3])
            u_s.append(u)
            accs.append(acc)
        if "' " in line_strs[1]:
            class_accs[line_strs[1].replace("'", "").replace(" ", "")].append(float(line_strs[2].split(" ")[0]))
            # print(line_strs[1].replace(" ", ""), float(line_strs[2].split(" ")[0]))
        # if line_strs[1] == "test":
        #     accs.append(float(line_strs[4]))
    # class_accs["u"] = u_s
    # class_accs["acc"] = accs
    with open('visualization/c3d.csv', mode='w', newline='\n') as csv_file:
        data = np.array([u_s] + list(class_accs.values()) + [accs])
        print(data.shape)
        fieldnames = ["u"] + list(class_accs.keys()) + ["Average Accuracy"]
        writer = csv.writer(csv_file, delimiter='	')

        writer.writerow(fieldnames)
        # writer.writerow(u_s)
        # writer.writerow(accs)
        for i in range(data.shape[1]):
            writer.writerow(data[:, i])

def accuracy_u_plot():
    output_format = "pdf"
    data = pd.read_table("visualization/c3d.csv", comment="#")
    # data = pd.read_table("visualization/cifar10-densenet-indep-utilization.csv", comment="#")
    print(data.columns)
    data = pd.melt(data, id_vars=["u"])
    print(data)
    print(data.columns)
    xtics = [i / 16 for i in range(0, 17, 4)]
    palette = ["#41b6c4ff", "#2c7fb8ff", "#253494ff", "#000000ff"]
    plot = (ggplot(aes(x="u", y="value", color="variable"), data=data)
            + theme(text=element_text(family="serif", size=16))
            # + ggtitle("Non-Uniform Resource Allocation")
            + xlab("Utilization parameter u")
            + scale_x_continuous(breaks=xtics)
            # + theme( axis_text_x=element_text(size=6) )
            + ylab("Prediction accuracy")
            + scale_color_discrete(guide=False)
            # + theme(legend_position="right", legend_title=element_text(text=""),
            #         legend_text=element_text(size=4), legend_key_size=4)
            # + guides(color=guide_legend(ncol=1))
            + geom_point()
            + theme(axis_text_x = element_text(size=10), axis_text_y = element_text(size=10),
                    strip_text = element_text(size=7))
            + annotate("point", x=1.0, y=0.8267)
            + facet_wrap(['variable'], nrow=5)
            + geom_line(size=1.5)
            # + scale_size_manual(values=[20])
            # + scale_linetype_manual(values=["dashed"])
            # + geom_line(aes(group="acc", size=2))
            # + geom_point(aes(shape="variable"))
            # + scale_linetype_manual(name="variable", values=("solid", "solid", "solid", "dashed"))
            # + scale_shape_manual(name="variable", values=("o", "o", "o", "None"))
            # + scale_color_manual(name="variable", values=palette)
            )

    plot += annotate("point", x=1.0, y=0.8267, color="black", alpha=0.5)
    # + scale_color_brewer( type="seq", palette="Blues" ))
    plot.draw().show()
    # ggsave( plot, "cifar10-densenet-indep-utilization.pdf", device="pdf", width=5, height=3 )
    ggsave(plot, "visualization/acc_for_all.{}".format(output_format), width=12, height=8)

def gpu_plot():
    output_format = "pdf"
    title = ""
    df = pd.read_csv("visualization/xavier-power-results_post.csv", comment="#")
    df.set_index("u")
    # plt.style.use('')
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)
    fig.set_size_inches(12, 4)
    ax.set_axisbelow(True)
    ax.invert_xaxis()
    # ax.spines["left"].set_visible(True)
    ax.spines["left"].set_edgecolor('black')
    ax.spines["left"].set_linewidth(1)
    def make_patch_spines_invisible(ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)


    y2 = ax.twinx()
    y3 = ax.twinx()
    make_patch_spines_invisible(y2)
    make_patch_spines_invisible(y3)
    y2.grid(False)
    y3.grid(False)
    y3.spines["right"].set_position(("axes", 1.1))
    # make_patch_spines_invisible(y3)


    y2.spines["right"].set_visible(True)
    y3.spines["right"].set_visible(True)
    y2.spines["right"].set_edgecolor('black')
    y3.spines["right"].set_edgecolor('black')



    # tkw = dict(size=4, width=1.5)
    # y3.tick_params(axis='y', **tkw)
    acc_df = df[df["facet"] == "accuracy"]
    ax.plot(acc_df["u"], acc_df["mean"], color="r", marker="o", markersize=7, linewidth=4, alpha=0.5)
    ax.fill_between(acc_df["u"], acc_df["se_lower"], acc_df["se_upper"], facecolor="silver")
    ax.annotate("Accuracy", (0.65, 0.82), xycoords='figure fraction', fontsize=16, color="black", bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    # power
    power_df = df[df["facet"] == "power"]
    y2.plot(power_df["u"], power_df["mean"], color="b", marker="o", markersize=7, linewidth=4, alpha=0.5)
    y2.fill_between(power_df["u"], power_df["se_lower"], power_df["se_upper"], facecolor="silver", edgecolor="gray")
    y2.annotate("Mean Power", (0.57, 0.63), xycoords='figure fraction', fontsize=16, color="black",
                bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    # # time
    time_df = df[df["facet"] == "time"]
    y3.plot(time_df["u"], time_df["mean"], color="g", marker="o", markersize=7, linewidth=4, alpha=0.5)
    y3.fill_between(time_df["u"], time_df["se_lower"], time_df["se_upper"], facecolor="silver", edgecolor="gray")
    y3.annotate("Run Time", (0.39, 0.47), xycoords='figure fraction', fontsize=16, color="black",
                bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 5})
    ax.set_xlabel("Utilization parameter u", fontsize=20, color='black')
    ax.set_ylabel("Accuracy", fontsize=20, color='black')
    y2.set_ylabel("Power (W)", fontsize=20, color='black')
    y3.set_ylabel("Time (S)", fontsize=20, color='black')

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    y2.yaxis.set_tick_params(labelsize=20)
    y3.yaxis.set_tick_params(labelsize=20)

    fig.show()
    fig.savefig('visualization/gpu_plot.pdf', pad_inches=0)

def plot_xavier(filename, title):
    format = 'pdf'
    xtics = [i / 16 for i in range(0, 17, 4)]
    # palette = ['#a6cee3ff','#1f78b4ff','#b2df8aff','#33a02cff','#fb9a99ff','#e31a1cff']
    palette = ['#e41a1cff', '#377eb8ff', '#4daf4aff', '#984ea3ff', '#ff7f00ff', '#ffff33ff']

    xavier_height = 3.5
    data = pd.read_csv("{}.csv".format(filename), comment="#")
    def labeller(name):
        return {
            "accuracy": "Accuracy",
            "power": "Mean Power (W)",
            "time": "Run Time (s)"
        }[name]

    print(xtics)

    plot = (ggplot(aes(x="u", y="mean", color="facet"), data=data)
            + theme(text=element_text(family="serif"))
            + ggtitle(title)
            + xlab("Throttle Setting")
            # + scale_x_continuous( breaks=xtics )
            + scale_x_reverse(breaks=([-0.25] + xtics))
            # + theme( axis_text_x=element_text(size=6) )
            # + ylim( (0, 0.95) )
            + ylab("")
            + geom_line(show_legend=False)  # aes(linetype="algorithm"), show_legend=legend )
            + geom_point(show_legend=False)  # aes(shape="algorithm"), show_legend=legend )
            + geom_ribbon(aes(x="u", ymin="se_lower",
                              ymax="se_upper"),
                          inherit_aes=False, alpha=0.2)
            + scale_color_brewer(type="qual", palette="Set1")
            # + scale_shape( palette=manual_pal([
            # # Filled
            # 'o',  # circle
            # '^',  # triangle up
            # 's',  # square
            # 'D',  # Diamond
            # 'v'  # triangle down
            # ]) )
            # + facet_grid(["facet", "."], scales="free_y",
            #              labeller=labeller)
            + theme(panel_spacing=0.1)
            + theme(plot_title=element_text(lineheight=1.5))
            )
    # legend_title=element_text(text="Gating method") )
    # plot += guides( color=guide_legend(nrow=2) )
    # ggsave( plot, "{}.pdf".format(filename), device="pdf", width=12, height=3.5 )
    ggsave(plot, "{}.{}".format(filename, format), width=5, height=6.5)



if __name__ == "__main__":
    logs = load_logs("logs/eval_raw_c3d_2019-11-07_17-28-30.json")
    results = process_logs(logs)
    # print(results)
    u_vs_examples_plot(results)
    # load_acc_log()
    # accuracy_u_plot()
    # gpu_plot()
    # plot_xavier("visualization/xavier-power-results_post", "Throttling Performance -- VGG-D / CIFAR10 \non NVIDIA Jetson AGX Xavier")