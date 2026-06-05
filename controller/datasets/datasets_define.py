import os
import glob
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class EpisodePairDataset(Dataset):
    def __init__(self, file_paths, max_step_dist=15, stop_step_dist=5, num_classes=3, balanced=True):
        ''' 数据集导入
        :param file_paths: 每个数据集为一集(主目录/episode_*.pt)
        :param max_step_dist: 两个特征的间隔不超过<最大步长>
        :param stop_step_dist: 输入两个特征的时间步长小于<停止步长>时认为是已经达到附近
        :param num_classes: 动作种类,用于one-hot
        '''
        self.max_step_dist = max_step_dist
        self.stop_step_dist = stop_step_dist
        self.num_classes = num_classes
        self.file_paths = file_paths

        done_true_indices = []      # step_dist < stop_step_dist
        done_false_indices = []     # stop_step_dist <= step_dist < max_step_dist

        for path in self.file_paths:
            data = torch.load(path, map_location="cpu")
            embeddings = data["embeddings"]
            seq_len = embeddings.size(0)
            for t in range(seq_len):
                for i in range(1, max_step_dist):
                    if t + i < seq_len:
                        sample_tuple = (path, t, t + i)
                        if i < self.stop_step_dist:
                            done_true_indices.append(sample_tuple)
                        else:
                            done_false_indices.append(sample_tuple)
        # 平衡done正负样本
        num_true = len(done_true_indices)
        num_false = len(done_false_indices)
        sample_num = min(num_true, num_false)
        if balanced:
            sampled_true = random.sample(done_true_indices, sample_num)
            sampled_false = random.sample(done_false_indices, sample_num)
            self.samples_index = sampled_true + sampled_false
            random.shuffle(self.samples_index)
        else:
            self.samples_index = done_true_indices + done_false_indices

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        path, t, t_next = self.samples_index[idx]
        data = torch.load(path, map_location="cpu")
        embeddings = data["embeddings"]
        actions = data["actions"]
        e_t = embeddings[t]
        e_next = embeddings[t_next]
        feat_pair = torch.stack([e_t, e_next], dim=0)  # [2, 768]

        l = t_next - t
        done_val = 1.0 if l < self.stop_step_dist else 0.0
        done = torch.tensor([done_val], dtype=torch.float32)

        a_t = actions[t]
        if isinstance(a_t, torch.Tensor):
            a_t = a_t.item()
        action_one_hot = F.one_hot(torch.tensor(a_t), num_classes=self.num_classes).to(torch.float32)  # [3]

        return feat_pair, action_one_hot, done


if __name__ == "__main__":
    DATA_DIR = "./collected_data"
    all_datasets = glob.glob(os.path.join(DATA_DIR, "episode_*.pt"))

    dataset = EpisodePairDataset(
        file_paths=all_datasets,
        max_step_dist=15,
        stop_step_dist=5,
        num_classes=3
    )

    if len(dataset) > 0:
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for batch_feat_pairs, batch_actions, batch_dones in dataloader:
            print(f"特征对形状 (batch, 2, feat_dim): {batch_feat_pairs.shape}")
            print(f"action形状 (batch, 3):          {batch_actions.shape}")
            print(f"done形状 (batch, 1):       {batch_dones.shape}")
            print(f"action: {batch_actions[0]}")
            print(f"done:    {batch_dones[0].item()}")
            break
    else:
        print(f"未在 '{DATA_DIR}' 目录中找到符合格式的 'episode_*.pt' 文件，请检查路径。")
