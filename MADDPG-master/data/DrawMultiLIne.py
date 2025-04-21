#  checkTrainingData
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    # 坐标轴留白
    'axes.xmargin': 0.05,    # X轴右侧留白5%
    'axes.ymargin': 0.05,    # Y轴顶部留白5%

    # 字体和标签
    'font.family': 'Arial',  # 默认字体
    'font.size': 12,         # 全局字体大小
    'axes.labelsize': 12,    # 坐标轴标签字体大小
    'xtick.labelsize': 10,   # X轴刻度字体
    'ytick.labelsize': 10,   # Y轴刻度字体
})

def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def cumulative_mean(data):
    return np.cumsum(data) / (np.arange(len(data)) + 1)


FONT = "Arial"


def plotGraph(scheme=1):
    EpiAO = "Epi_Agent_obs_steps.npy"
    EpiR = "Epi_Reward.npy"
    EpiC = "Epi_Collisions.npy"
    EpiS = "Epi_Success_Steps.npy"
    EpiTO = "Epi_Target_beObserved_steps.npy"
    T_EpiC = "Test_Epi_collisions.npy"
    EpiSR = "Epi_Success_Rate.npy"

    if scheme == 1:
        pathIQL = "D:\\pycharmExtraProj\\MADDPG-MPE-copy\\MADDPG-master\\data\\fto_assignTargets\\MASAC"

        pathMADDPG = "D:\\pycharmExtraProj\\MADDPG-MPE-copy\\MADDPG-master\\data\\fto_assignTargets\\MADDPG"

        pathMADDPGS = "D:\\pycharmExtraProj\\MADDPG-MPE-copy\\MADDPG-master\\data\\fto_assignTargets\\MADDPG_soft"

        pathMADDPGSN = "D:\\pycharmExtraProj\\MADDPG-MPE-copy\\MADDPG-master\\data\\fto_assignTargets\\MADDPG_SAFENET"

        pathMASACSN = "D:\pycharmExtraProj\MADDPG-MPE-copy\MADDPG-master\data\\fto_assignTargets\MASAC_soft"




        IQL_ER = np.load(pathIQL + '/' + EpiR)
        IQL_EC = np.load(pathIQL + '/' + EpiC)
        IQL_TEC = np.load(pathIQL + '/' + T_EpiC)
        IQL_ESR = np.load(pathIQL + '/' + EpiSR)
        IQL_ESR = np.insert(IQL_ESR, 0, 0)

        maddpg_ER = np.load(pathMADDPG + '/' + EpiR)
        maddpg_EC = np.load(pathMADDPG + '/' + EpiC)
        maddpg_TEC = np.load(pathMADDPG + '/' + T_EpiC)
        maddpg_ESR = np.load(pathMADDPG + '/' + EpiSR)
        maddpg_ESR = np.insert(maddpg_ESR, 0, 0)

        maddpgs_ER = np.load(pathMADDPGS + '/' + EpiR)
        maddpgs_EC = np.load(pathMADDPGS + '/' + EpiC)
        maddpgs_TEC = np.load(pathMADDPGS + '/' + T_EpiC)
        maddpgs_ESR = np.load(pathMADDPGS + '/' + EpiSR)
        maddpgs_ESR = np.insert(maddpgs_ESR, 0, 0)

        masacsn_ER = np.load(pathMASACSN + '/' + EpiR)
        masacsn_EC = np.load(pathMASACSN + '/' + EpiC)
        masacsn_TEC = np.load(pathMASACSN + '/' + T_EpiC)
        masacsn_ESR = np.load(pathMASACSN + '/' + EpiSR)
        masacsn_ESR = np.insert(masacsn_ESR, 0, 0)

        maddpgsn_TEC = np.load(pathMADDPGSN + '/' + T_EpiC)


        #
        print(IQL_ESR[-1])
        print(maddpg_ESR[-1])
        print(maddpgs_ESR[-1])
        # 滑动平均
        IQL_TEC_ma = moving_average(IQL_TEC)
        maddpg_TEC_ma = moving_average(maddpg_TEC)
        maddpgsn_TEC_ma = moving_average(maddpgsn_TEC)

        IQL_ER_ma = moving_average(IQL_ER)
        maddpg_ER_ma = moving_average(maddpg_ER)
        maddpgs_ER_ma = moving_average(maddpgs_ER)

        # 实际平均
        IQL_TEC_cummean = cumulative_mean(IQL_TEC / 2)
        maddpg_TEC_cummean = cumulative_mean(maddpg_TEC / 2)
        maddpgsn_TEC_cummean = cumulative_mean(maddpgsn_TEC / 2)
        masacsn_TEC_cummean = cumulative_mean(masacsn_TEC / 2)

        print("average collisions test: masac, maddpg, maddpgsn:")
        print(IQL_TEC_cummean[-1])
        print(maddpg_TEC_cummean[-1])
        print(maddpgsn_TEC_cummean[-1])
        # REWARD CURVE
        plt.figure()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        # plt.plot(range(1, len(IQL_ER_ma)+1), IQL_ER_ma, label="MASAC-RS", alpha=0.8, linewidth=2)
        # plt.plot(range(1, len(maddpg_ER_ma)+1), maddpg_ER_ma, label="MADDPG-RS", alpha=0.8, linewidth=2)
        plt.plot(range(1, len(maddpgs_ER)+1), maddpgs_ER, label="MTT-SN", alpha=0.8, linewidth=2)
        plt.legend()
        plt.xlim(0, 1500)
        plt.savefig("A_combine_reward.png", dpi=300)

        # Success Rate
        plt.figure()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel("Timesteps * 10000",
                   fontsize=12,  # 字体大小（通常10-14pt）

                   fontfamily=FONT)  # 字体（推荐Arial/Times New Roman）

        plt.ylabel("Training Success Rate",
                   fontsize=12,  # 字体大小（通常10-14pt）

                   fontfamily=FONT)  # 字体（推荐Arial/Times New Roman）

        plt.plot(range(len(IQL_ESR)), IQL_ESR,
                 label="MASAC-RS",
                 color='green',
                 linestyle='-',
                 linewidth=1.5,
                 marker='s',
                 alpha=0.9
                 )
        plt.plot(range(len(maddpg_ESR)), maddpg_ESR,
                 label="MADDPG-RS",
                 color='blue',
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.9,
                 marker='^')
        plt.plot(range(len(maddpgs_ESR)), maddpgs_ESR,
                 label="MTT-SN",
                 color='red',
                 linestyle='-',
                 linewidth=1.5,
                 marker='o',
                 markersize=6,
                 alpha=0.9
                 )
        plt.plot(range(len(masacsn_ESR)), masacsn_ESR,
                label="MASAC-SN",
                color='brown',
                linestyle='-',
                linewidth=1.5,
                marker='o',
                markersize=6,
                alpha=0.9
                )
        plt.legend(
            fontsize=12,  # 字体大小
            prop={'family': FONT, 'size': 12},  # 详细字体属性)
        )
        # xls = range(len(IQL_ESR))
        # max_x = xls[-1]
        # xticks = list(plt.xticks()[0])
        # if max_x not in xticks:
        #     plt.xticks(xticks + [max_x])
        # plt.xlim(0, 1500)

        plt.savefig("A_combine_training_success_rate.png", dpi=300)

        # collision in train
        plt.figure()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel("Episode")
        plt.ylabel("Total collisions")
        plt.plot(range(len(IQL_EC)), [np.sum(IQL_EC[:i] / 2) for i in range(len(IQL_EC))],
                 label="MASAC-RS", color='green',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9)
        plt.plot(range(len(maddpg_EC)), [np.sum(maddpg_EC[:i] / 2) for i in range(len(maddpg_EC))],
                 label="MADDPG-RS",
                 color='blue',
                 linestyle='-',
                 linewidth=1.5,


                 alpha=0.9)
        plt.plot(range(len(maddpgs_EC)), [np.sum(maddpgs_EC[:i] / 2) for i in range(len(maddpgs_EC))],
                 label="MTT-SN",
                 color='red',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9
                 )

        plt.plot(range(len(masacsn_EC)), [np.sum(masacsn_EC[:i] / 2) for i in range(len(masacsn_EC))],
                 label="MASAC-SN",
                 color='brown',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9
                 )

        plt.legend(
            fontsize=12,  # 字体大小
            prop={'family': FONT, 'size': 12},  # 详细字体属性)
        )
        plt.xlim(0, 1500)
        plt.savefig("A_combine_collision.png", dpi=300)


        print("collisionCMP: sn:maddpg->" + str(np.sum(maddpgs_TEC) / np.sum(maddpg_TEC)))
        print("collisionCMP: sn:masac->" + str(np.sum(maddpgs_TEC) / np.sum(IQL_TEC)))
        print("collisions:sn, maddpg, masac" + str(np.sum(maddpgs_TEC)/2))
        print(np.sum(maddpg_TEC) / 2)
        print(np.sum(IQL_TEC) / 2)
        print("collision:masacsn," + str(np.sum(masacsn_TEC)/2))

        plt.figure()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel("Timesteps")
        plt.ylabel("Cumulative collisions in evaluation")
        plt.plot(np.array(range(len(IQL_TEC))) * 200, [np.sum(IQL_TEC[:i] / 2) for i in range(len(IQL_TEC))],
                 label="MASAC-RS",
                 color='green',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9
                 )
        plt.plot(np.array(range(len(masacsn_TEC))) * 200, [np.sum(masacsn_TEC[:i] / 2) for i in range(len(masacsn_TEC))],
                 label="MASAC-SN",
                 color='brown',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9
                 )
        plt.plot(np.array(range(len(maddpg_TEC))) * 200, [np.sum(maddpg_TEC[:i] / 2) for i in range(len(maddpg_TEC))],
                 label="MADDPG-RS",
                 color='blue',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9)
        # plt.plot(range(len(maddpgs_TEC)), [np.sum(maddpgs_TEC[:i]/2) for i in range(len(maddpgs_TEC))], label="MADDPG_SF")
        plt.plot(np.array(range(len(maddpgsn_TEC))) * 200,
                 [np.sum(maddpgsn_TEC[:i] / 2) for i in range(len(maddpgsn_TEC))],
                 label="MTT-SN",
                 color='red',
                 linestyle='-',
                 linewidth=1.5,

                 alpha=0.9)
        plt.legend(
            fontsize=12,  # 字体大小
            prop={'family': FONT, 'size': 12},  # 详细字体属性)
        )
        plt.ylim(0, 10000)
        plt.xlim(0, 200000)
        plt.tight_layout()
        plt.savefig("A_combine_eval_collision.png", dpi=300)


        plt.figure()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.xlabel("Episodes")
        plt.ylabel("Average collisions for each episode")
        plt.plot(np.array(range(len(IQL_TEC_cummean))), IQL_TEC_cummean,
                 label="MASAC-RS",
                 color='green',
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.9
                 )
        plt.plot(np.array(range(len(maddpg_TEC_cummean))), maddpg_TEC_cummean,
                 label="MADDPG-RS",
                 color='blue',
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.9
                 )
        # plt.plot(range(len(maddpgs_TEC)), [np.sum(maddpgs_TEC[:i]/2) for i in range(len(maddpgs_TEC))], label="MADDPG_SF")
        plt.plot(np.array(range(len(maddpgsn_TEC_cummean))), maddpgsn_TEC_cummean, label="MTT-SN",
                 color='red',
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.9
                 )
        plt.plot(np.array(range(len(masacsn_TEC_cummean))), masacsn_TEC_cummean, label="MASAC-SN",
                 color='brown',
                 linestyle='-',
                 linewidth=1.5,
                 alpha=0.9
                 )
        plt.legend(
            fontsize=12,  # 字体大小
            prop={'family': FONT, 'size': 12},  # 详细字体属性)
        )
        plt.tight_layout()
        plt.savefig("A_combine_eval_collision_per_epi.png", dpi=300)


plotGraph()
