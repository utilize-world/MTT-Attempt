#  checkTrainingData
import numpy as np
import matplotlib.pyplot as plt


def plotGraph(path, name, scheme=0, test=False):
    EpiAO = "Epi_Agent_obs_steps.npy"
    EpiR = "Epi_Reward.npy"
    EpiC = "Epi_Collisions.npy"
    EpiS = "Epi_Success_Steps.npy"
    EpiTO = "Epi_Target_beObserved_steps.npy"
    if test:
        T_R = "Test_Epi_Reward.npy"
        T_C = "Test_Epi_collisions.npy"
        T_CR = "Test_Epi_collisionRate.npy"
    data_AO = np.load(path + '/' + EpiAO)
    data_R = np.load(path + '/' + EpiR)
    data_C = np.load(path + '/' + EpiC)
    data_S = np.load(path + '/' + EpiS)
    data_TO = np.load(path + '/' + EpiTO)
    if test:
        T_data_R = np.load(path + '/' + T_R)
        T_data_C = np.load(path + '/' + T_C)
        T_data_CR = np.load(path + '/' + T_CR)
    # if scheme==0:
    #     data_i = np.load(path + '/' + infeasible)

    # print("ave reward=" + str(np.mean(data_r[-200:])))

    fig = plt.figure()
    ax1 = fig.add_subplot(321)

    ax1.plot(range(len(data_R)), data_R, label="Epi_Reward")
    plt.xlabel("epi")
    plt.ylabel("reward")

    ax2 = fig.add_subplot(322)

    ax2.plot(range(len(data_C)), data_C, label="collsions")
    ax2.legend()
    plt.xlabel("epi")
    plt.ylabel("collisions")

    if scheme == 0:
        ax3 = fig.add_subplot(323)

        ax3.plot(range(len(data_S)), data_S, label="SuccessStep")
        ax3.legend()
        plt.xlabel("epi")
        plt.ylabel("SuccessStep")

    ax4 = fig.add_subplot(324)

    ax4.plot(range(len(data_C)), [np.sum(data_C[:i]) for i in range(len(data_C))], label="cumulate collsions")
    ax4.legend()
    plt.xlabel("epi")
    plt.ylabel("cumulate collisions")

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    plt.savefig(name + ".png")
    plt.close()

    if test:
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(411)
        ax1.plot(range(len(T_data_R)), T_data_R, label="Test_Epi_Reward")

        plt.xlabel("epi")
        plt.ylabel("evl_reward")

        ax2 = fig.add_subplot(412)
        ax2.plot(range(len(T_data_C)), T_data_C / 2, label="Test_Epi_Collision")

        plt.xlabel("epi")
        plt.ylabel("evl_collisions")

        ax3 = fig.add_subplot(413)
        ax3.plot(range(len(T_data_CR)), T_data_CR, label="Test_Epi_CollisionRate")
        plt.xlabel("epi")
        plt.ylabel("evl_collisionRate")

        ax4 = fig.add_subplot(414)
        ax4.plot(range(len(T_data_C)), [np.sum(T_data_C[:i] / 2) for i in range(len(T_data_C))], label="cumulate collsions")
        plt.xlabel("epi")
        plt.ylabel("evl_cumulative_collisionRate")

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4)

        plt.savefig("Test_" + name + ".png")
        plt.close()


def plot1(path):
    data_name = "test_collisions.npy"

    name = ["SafeMADDPG_soft", "SafeMADDPG_hard", "MADDPG"]
    data = []
    for i, _path in enumerate(path):
        data.append(np.load(_path + '/' + data_name))

    plt.figure()
    for i, _data in enumerate(data):
        plt.plot(range(len(_data)), _data, label=name[i])
    plt.xlabel("epi")
    plt.ylabel("test_collisions")
    plt.legend()
    plt.savefig("test_collisions.png")
    plt.close()
    for i, _data in enumerate(data):
        print(name[i] + "-->ave test_collisions=" + str(np.mean(_data)))


if __name__ == '__main__':
    #path = "D:\\pycharmExtraProj\\MADDPG-MPE-copy\\MADDPG-master\\data\\fto_assignTargets\\MADDPG"
    path = "D:\\pycharmExtraProj\\MADDPG-MPE-copy\\MADDPG-master\\data\\fto_assignTargets\\MADDPG_soft"
    name = ""
    path += name
    # plotGraph(path, "maddpg" + name + "fto-5v5-sf", test=True)
    plotGraph(path, "MADDPG" + name + "MADDPG-individualReward-soft", test=True)
    # path2 = "E:/chromeDownload/deep-rl-main/data/agents/MADDPG/seed_dist"
    # path3 = "E:/chromeDownload/deep-rl-main/data/agents/SafeMADDPG_hard/seed_dist"
    # plotGraph(path2, "MADDPG", 1)
    # plotGraph(path3, "SafeMADDPG_hard")
    # test_plot([path, path3, path2])
