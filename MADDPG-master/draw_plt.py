import matplotlib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import csv


# # 以下是测试画图的代码
# def clear_csv(filepath, data_df):
#     open(filepath, 'w').close()
#     data_df.to_csv(filepath, index=False, mode='a')
#
#
# matplotlib.use('Qt5Agg')  # 启用交互式后端
# # test
# input_file_path = "data_p"
# save_fig_p = 'figures'
# input_file_path = os.path.join(input_file_path, 'output_data.csv')
# fig_save_path = os.path.join('figures', 'output.png')
# # df = pd.read_csv(input_file_path)
# output_path = input_file_path
# data1 = [i for i in range(10)]
# data2 = [(i + np.random.rand()) for i in range(10, 20)]
# data = {'row1': data1, 'row2': data2}
# data_df = pd.DataFrame(data)
#
# print(data_df)
#
# df = pd.read_csv(input_file_path)
# if len(df) > 100:
#     clear_csv(output_path, data_df)
# length = len(df)
#
#
# while True:
#     data1 = [i for i in range(10)]
#     data2 = [(i + 10 * np.random.rand()) for i in range(10, 20)]
#     data = {'row1': data1, 'row2': data2}
#     data_df = pd.DataFrame(data)
#     data_df.to_csv(output_path, index=False, mode='a', header=False)
#     df = pd.read_csv(input_file_path)
#     length = len(df)
#     if length > 100:
#         break
#
# # ds = df.sort_values(by='row1')    # 排序，实际上没必要排序，sns.lineplot会自动检索相同的x，对应多个y
# # print(ds)
# sns.lineplot(data=df, x='row1', y='row2', ci=95)
# plt.savefig(fig_save_path, format='png')
# plt.show()
# 定义一个全套流程的画图与数据处理函数
def collect_data_and_save_drawings(data, index, name, read_path, save_fig_path, csv_path, average_count=10):
    """
    data: 初始数据
    index: 定义的训练总次数
    name: 算法名字
    read_path: 读取csv文件的父目录
    save_fig_path: 保存图像的路径
    csv_path: 保存csv文件的父目录
    average_count: 用来控制平均的episode数，默认为10，即计算十个episodes训练时，每个episode的平均奖励
    """
    # 每隔一定数量的episode就画一次图,否则
    x_name = 'Episodes'
    y_name = 'Average Episode Reward'
    data_T = training_data_to_dataFrame(data, [x_name, y_name])
    save_data_to_csv(data_T, csv_path, name, index)

    if index % average_count == 0:
        df = data_process(name, read_path)
        df = df.reset_index(drop=True)   # 这一步重置index   没什么必要
        #   df.sort_values(by=x_name)    # 对x轴进行排序，同样没必要，在snsplot的时候自动排列
        name = name + str(index)
        # print(df)
        draw_sns_plot(dataFrame=df, x_name=x_name, y_name=y_name, save_path=save_fig_path, name=name)
        return True

    return False


def draw_sns_plot(dataFrame, x_name, y_name, save_path, name, ci=95):
    """
    用来画带阴影的平均奖励图，第一项代表dataFrame的格式的数据，第二,三项代表要画的图的横坐标和纵坐标名字，ci代表置信区间
    """
    plt.figure()
    print(dataFrame)
    sns.lineplot(data=dataFrame, x=x_name, y=y_name, ci=ci, label='test')   # 这里的data绝对不能少，不然报错

    # 调整边距
    # plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
    plt.legend(fontsize=12)
    plt.title("Average Episode Reward")
    save_path = os.path.join(save_path, (name + '.png'))
    plt.savefig(save_path, dpi=300)


def save_data_to_csv(data, csv_path, name, index):
    """
    存放的是dataFrame格式的数据, name表示算法名字，index代表第几次训练
    """
    name = name + "_" + str(index)
    save_file_path = os.path.join(csv_path, (name + '.csv'))
    # 保存格式是 csvpath(父目录) + 算法名字_第几次训练 + .png
    # 以下主要是为了用第一行数据导入表头进csv,并刷新该文件的内容
    dt1 = data.loc[0]
    dt2 = data.loc[1:]
    data.to_csv(save_file_path, index=False, mode='w', header=True)



def training_data_to_dataFrame(data, row_name, dimension=2):
    """
    row_name是一个字符串数组，代表数据的维度，这里一般都是默认二维
    """
    if dimension is not 2:
        print("implement error")
    r_1_name = row_name[0]
    r_2_name = row_name[1]
    data_1 = [i for i in range(len(data))]
    data_2 = data
    data_T = {r_1_name: data_1, r_2_name: data_2}
    dataFrame = pd.DataFrame(data_T)
    return dataFrame


def data_process(name, read_path):
    """
    根据训练的个数，读取所有的csv文件并拼接在一起，为画图提供总的dataFrame,这里的readpath依然是父目录，然后遍历该目录下所有
    """
    df = pd.DataFrame()
    for root, dirs, files in os.walk(read_path):
        for file in files:
            if name in file:
                file_path = os.path.join(read_path, file)
                df = pd.concat([df, pd.read_csv(file_path)])    # 将所有的csv文件中的数据全部叠在一起
    return df