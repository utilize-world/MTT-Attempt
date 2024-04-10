from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
from draw_plt import collect_data_and_save_drawings
from utils import clear_folder
seed = 0

if __name__ == '__main__':
    # get the params
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = get_args()
    index = args.training_times    # 训练次数
    print(args)
    print(type(args))
    env, args = make_env(args)  # 将一系列输入的参数和环境作为变量，传递给下面的运行
    runner = Runner(args, env)  #
    clear_folder(runner.csv_save_dir)   # 清空目标文件夹中的所有文件

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        for i in range(1, index + 1):
            print("running")
            data = runner.run()
            collect_data_and_save_drawings(data=data,
                                           index=i,
                                           name="MADDPG",
                                           read_path=runner.csv_save_dir,
                                           save_fig_path=runner.fig_save_dir,
                                           csv_path=runner.csv_save_dir,
                                           average_count=10
                                           )