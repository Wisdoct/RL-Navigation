import glob
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from controller.datasets.datasets_define import EpisodePairDataset
from controller.model.controller import MLPController
from controller.train import AttentionControllerShortcut

MAX_STEP_DIST = 10
STOP_DIST = 2

if __name__ == '__main__':
    # data_dir = r".\datasets\record_real"
    data_dir = r".\datasets\record_sim"
    val_ratio = 0.2
    batch_size = 64
    epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    all_files = glob.glob(os.path.join(data_dir, "episode_*.pt"))
    # all_files = glob.glob(os.path.join(data_dir, "episode_*_step_*.pt"))
    random.shuffle(all_files)
    val_size = int(len(all_files) * val_ratio)
    val_files = all_files[:val_size]
    train_files = all_files[val_size:]

    val_dataset = EpisodePairDataset(val_files, max_step_dist=MAX_STEP_DIST, stop_step_dist=STOP_DIST)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = AttentionControllerShortcut(feat_dim=768, hidden_dim=512, m_dim=16, num_actions=3).to(device)
    # model = MLPController(feat_dim=768, hidden_dim=2048, num_actions=3).to(device)
    model.load_state_dict(torch.load('weights/new_nav_model.pth'))

    criterion_action = nn.CrossEntropyLoss()
    criterion_done = nn.BCEWithLogitsLoss()

    model.eval()
    val_loss = 0.0
    val_act_correct = 0
    val_done_correct = 0
    total_val_samples = 0

    with torch.no_grad():
        for feat_pairs, actions_one_hot, dones in val_loader:
            feat_pairs = feat_pairs.to(device)
            actions_one_hot = actions_one_hot.to(device)
            dones = dones.to(device)

            action_labels = torch.argmax(actions_one_hot, dim=1)

            action_logits, done_logits = model(feat_pairs)

            loss_action = criterion_action(action_logits, action_labels)
            loss_done = criterion_done(done_logits, dones)
            loss = loss_action + 3 * loss_done

            val_loss += loss.item() * feat_pairs.size(0)

            act_preds = torch.argmax(action_logits, dim=1)
            val_act_correct += (act_preds == action_labels).sum().item()

            done_preds = (torch.sigmoid(done_logits) > 0.5).float()
            val_done_correct += (done_preds == dones).sum().item()

            total_val_samples += feat_pairs.size(0)

    epoch_val_loss = val_loss / total_val_samples
    epoch_val_act_acc = val_act_correct / total_val_samples
    epoch_val_done_acc = val_done_correct / total_val_samples

    # 打印当前 Epoch 的结果
    print(f"Val Loss: {epoch_val_loss:.4f} | Val Act Acc: {epoch_val_act_acc * 100:.2f}% | Val Done Acc: {epoch_val_done_acc * 100:.2f}%")
