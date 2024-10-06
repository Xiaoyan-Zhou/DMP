import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_DMPL_ID():
    file_path = r'E:\1-PHD\01ZXY\zxy24\Multi-prototype\0-DMPL-paper\00-results_paper.xlsx'  # 修改为你的Excel文件路径
    df = pd.read_excel(file_path, sheet_name='ID_LAMBDA_K')

    # df = pd.DataFrame(data)

    # 创建透视表用于热图
    heatmap_data = df.pivot(index="Total number of prototype", columns="lambda", values="ACC")

    # 绘制热图
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    # plt.title("Heatmap of ACC by Lambda and Total Number of Prototypes", fontsize=18)
    # 修改 x 和 y 轴标签的字体大小
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.ylabel(r"$K$", fontsize=16)
    plt.savefig('./results/heatmap_ID_ACC_tau0.9.jpg', dpi=300)
    plt.show()

    # 为每组 prototype 绘制折线图
    plt.figure(figsize=(14, 8))
    for prototype in df['Total number of prototype'].unique():
        subset = df[df['Total number of prototype'] == prototype]
        plt.plot(subset['lambda'], subset['ACC'], marker='o', label=f'{prototype} prototypes')

    # plt.title("ACC vs Lambda for Different Numbers of Prototypes")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("ACC")
    plt.legend(title="Total number of prototypes")
    plt.grid(True)
    plt.show()


    # 3D 表面图
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    X = df['lambda']
    Y = df['Total number of prototype']
    Z = df['ACC']

    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title("3D Surface Plot of ACC vs Lambda and Total Number of Prototypes")
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Total number of prototypes")
    ax.set_zlabel("ACC")

    plt.savefig('./results/3D_ACC_tau0.9.jpg', dpi=300)
    plt.show()


def draw_DMPL_OOD(metric='AUROC'):
    file_path = r'E:\1-PHD\01ZXY\zxy24\Multi-prototype\0-DMPL-paper\00-results_paper.xlsx'  # 修改为你的Excel文件路径
    df = pd.read_excel(file_path, sheet_name='OOD_lambda_K')

    # 创建透视表用于热图
    heatmap_data = df.pivot(index="Total number of prototype", columns="lambda", values=metric)

    # 绘制热图
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
    # plt.title("Heatmap of ACC by Lambda and Total Number of Prototypes", fontsize=18)
    # 修改 x 和 y 轴标签的字体大小
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.ylabel(r"$K$", fontsize=16)
    plt.savefig('./results/heatmap_OOD_{}.jpg'.format(metric), dpi=300)
    plt.show()

    # 为每组 prototype 绘制折线图
    plt.figure(figsize=(14, 8))
    for prototype in df['Total number of prototype'].unique():
        subset = df[df['Total number of prototype'] == prototype]
        plt.plot(subset['lambda'], subset[metric], marker='o', label=f'{prototype} prototypes')

    plt.title("{} vs Lambda for Different Numbers of Prototypes".format(metric))
    plt.xlabel("Lambda")
    plt.ylabel(metric)
    plt.legend(title="Total number of prototypes")
    plt.grid(True)
    plt.savefig('./results/Line_OOD_{}.jpg'.format(metric), dpi=300)
    plt.show()

    # 3D 表面图
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    X = df['lambda'].values
    Y = df['Total number of prototype'].values
    Z = df[metric].values

    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_title("3D Surface Plot of {} vs Lambda and Total Number of Prototypes".format(metric))
    ax.set_xlabel(r"$\lambda$", fontsize=14, labelpad=12)
    ax.set_ylabel(r"$K$", fontsize=14)
    ax.set_zlabel(metric, labelpad=10, fontsize=14) # labelpad=20 bias, avoid the overlap of character and z axis

    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3])
    ax.set_xticklabels(ax.get_xticks(), rotation=45, ha='right', fontsize=10)

    plt.savefig('./results/3D_OOD_{}.jpg'.format(metric), dpi=300)
    plt.show()

def draw_uncertainty_filter(metric):
    file_path = r'E:\1-PHD\01ZXY\zxy24\Multi-prototype\0-DMPL-paper\00-results_paper.xlsx'   # 修改为你的Excel文件路径
    df = pd.read_excel(file_path, sheet_name='uncertainty filter')

    # 为每组 prototype 绘制折线图


    # 绘制 with mask 和 without mask 的折线图
    for prototype in df['Total number of prototype'].unique():
        plt.figure(figsize=(14, 8))
    # for prototype in [40]:
        subset = df[df['Total number of prototype'] == prototype]

        # 绘制 with mask 的线
        plt.plot(subset['lambda'], subset[metric+"(filter)"], marker='o', linestyle='-',
                 label=f'filter')

        # 绘制 without mask 的线
        plt.plot(subset['lambda'], subset[metric+"(no filter)"], marker='o', linestyle='-',
                 label=f'without filter')

        plt.title(f"{metric} vs"+r" $\lambda$ "+f"for {prototype} prototypes")
        plt.xlabel(r"$\lambda$")
        plt.ylabel(metric)
        plt.legend(title=r"uncertainty filter")
        plt.grid(True)
        plt.savefig(f'./results/Line_OOD_filter_{str(prototype)}_{metric}.jpg', dpi=300)
        plt.show()

def draw_threshold(metric):
    file_path = r'E:\1-PHD\01ZXY\zxy24\Multi-prototype\0-DMPL-paper\00-results_paper.xlsx'  # 修改为你的Excel文件路径
    df = pd.read_excel(file_path, sheet_name='threshold_tau_simple')

    # 创建透视表用于热图
    heatmap_data = df.pivot(index="Total number of prototype", columns=r"uncertainty threshold", values=metric)

    # 绘制热图
    plt.figure(figsize=(6, 7))
    if metric == 'ID ACC':
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
    else:
        sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
    # plt.title("Heatmap of ACC by Lambda and Total Number of Prototypes", fontsize=18)
    # 修改 x 和 y 轴标签的字体大小
    plt.xlabel(r"$\tau$", fontsize=16)
    plt.ylabel(r"$K$", fontsize=16)
    plt.savefig('./results/heatmap_tau_K_{}.jpg'.format(metric), dpi=300)
    plt.show()


    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    X = df['uncertainty threshold'].values
    Y = df['Total number of prototype'].values
    Z = df[metric].values

    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.set_title("3D Surface Plot of {} vs Lambda and Total Number of Prototypes".format(metric))
    ax.set_xlabel(r"$\tau$", fontsize=14, labelpad=12)
    ax.set_ylabel(r"$K$", fontsize=14)
    ax.set_zlabel(metric, labelpad=10, fontsize=14)  # labelpad=20 bias, avoid the overlap of character and z axis

    ax.set_xticks([0.85, 0.90, 0.95])
    ax.set_xticklabels(ax.get_xticks(), rotation=45, ha='right', fontsize=10)

    plt.savefig('./results/3D_tau_K_{}.jpg'.format(metric), dpi=300)
    plt.show()

if __name__ == '__main__':
    flag = 1 #0: draw the 3D map of lambda_number-of-prototype OOD 1: draw the compare with uncertainty mask or without uncertainty mask 2: draw the influence of threshold
    if flag ==0:
        draw_DMPL_OOD(metric="AUROC")
        draw_DMPL_OOD(metric="FPR")
    elif flag == 1:
        draw_uncertainty_filter(metric="AUROC")
        draw_uncertainty_filter(metric="FPR")
    elif flag == 2:
        draw_threshold(metric="FPR")
        draw_threshold(metric="AUROC")
        draw_threshold(metric="ID ACC")
    elif flag == 3: # draw ID acc heatmap using a deterministic uncertainty threshold
        draw_DMPL_ID()

