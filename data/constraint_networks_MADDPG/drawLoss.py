import numpy as np
from matplotlib import pyplot as plt

def draw(id):
    loss_training = np.load("constraint_net_loss_training_for_net_" + str(id) + ".npy")
    loss_evl = np.load("constraint_net_loss_validation_for_net_" + str(id) + ".npy")
    plt.figure()
    plt.plot(range(len(loss_training)), loss_training, label="training loss")

    plt.xlabel("epi")
    plt.ylabel("loss")
    plt.legend()

    plt.savefig("loss.png")
    plt.close()

    plt.figure()
    plt.plot(range(len(loss_evl)), loss_evl, label="evl loss")

    plt.xlabel("epi")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss_evl.png")


if __name__ == '__main__':
    draw(0)

