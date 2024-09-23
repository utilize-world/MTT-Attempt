from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return 1  # 因为我们每次返回整个批次的数据，所以长度为1

    def __getitem__(self, idx):
        return self.transitions