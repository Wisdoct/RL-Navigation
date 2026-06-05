import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==========================================
# 1. 继承并微调前面的 Dataset 适配文件划分
# ==========================================
# 稍微修改原 Dataset，使其支持传入预先划分好的文件列表，防止数据泄露
from torch.utils.data import Dataset
import torch.nn.functional as F

from controller.model.controller import MLPController, AttentionController, AttentionControllerShortcut
from datasets.datasets_define import EpisodePairDataset


MAX_STEP_DIST = 10
STOP_DIST = 2


def main():
    data_dir = "./datasets/record_sim"
    # data_dir = "./datasets/record_real"
    val_ratio = 0.2
    batch_size = 128
    epochs = 20
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    seed = 2
    random.seed(seed)
    torch.manual_seed(seed)

    all_files = glob.glob(os.path.join(data_dir, "episode_*.pt"))
    # all_files = glob.glob(os.path.join(data_dir, "scene_*.pt"))
    # print(all_files)

    random.shuffle(all_files)
    val_size = int(len(all_files) * val_ratio)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]
    # print(train_files)

    train_dataset = EpisodePairDataset(train_files, max_step_dist=MAX_STEP_DIST, stop_step_dist=STOP_DIST)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dataset = EpisodePairDataset(val_files, max_step_dist=MAX_STEP_DIST, stop_step_dist=STOP_DIST)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 初始化模型、损失函数与优化器
    model = AttentionControllerShortcut(feat_dim=768, hidden_dim=512, m_dim=16, num_actions=3).to(device)
    # model = AttentionController(feat_dim=768, hidden_dim=512, m_dim=16, num_actions=3).to(device)
    # model = MLPController(feat_dim=768, hidden_dim=2048, num_actions=3).to(device)
    # model.load_state_dict(torch.load('weights/best_nav_model.pth'))

    criterion_action = nn.CrossEntropyLoss()
    criterion_done = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    # train_losses = []
    # val_losses = []
    # train_act_acc = []
    # val_act_acc = []
    # train_done_acc = []
    # val_done_acc = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_act_correct = 0
        train_done_correct = 0
        num_train_samples = 0

        for feat_pairs, actions_one_hot, dones in train_loader:
            feat_pairs = feat_pairs.to(device)
            actions_one_hot = actions_one_hot.to(device)  # (batch, 3)
            dones = dones.to(device)  # (batch, 1)
            action_labels = torch.argmax(actions_one_hot, dim=1)

            optimizer.zero_grad()
            action_logits, done_logits = model(feat_pairs)
            loss_action = criterion_action(action_logits, action_labels)
            loss_done = criterion_done(done_logits, dones)
            # loss = loss_action + 2.0 * loss_done
            loss = loss_action + 3.0 * loss_done
            loss.backward()
            optimizer.step()

            # 训练损失
            train_loss += loss.item() * feat_pairs.size(0)
            act_preds = torch.argmax(action_logits, dim=1)
            train_act_correct += (act_preds == action_labels).sum().item()
            done_preds = (torch.sigmoid(done_logits) > 0.5).float()
            train_done_correct += (done_preds == dones).sum().item()
            num_train_samples += feat_pairs.size(0)

        epoch_train_loss = train_loss / num_train_samples
        epoch_train_act_acc = train_act_correct / num_train_samples
        epoch_train_done_acc = train_done_correct / num_train_samples


        model.eval()
        val_loss = 0
        val_act_correct = 0
        val_done_correct = 0
        num_val_samples = 0
        with torch.no_grad():
            for feat_pairs, actions_one_hot, dones in val_loader:
                feat_pairs = feat_pairs.to(device)
                actions_one_hot = actions_one_hot.to(device)
                dones = dones.to(device)
                action_labels = torch.argmax(actions_one_hot, dim=1)

                # 验证损失
                action_logits, done_logits = model(feat_pairs)
                loss_action = criterion_action(action_logits, action_labels)
                loss_done = criterion_done(done_logits, dones)
                loss = loss_action + 3.0 * loss_done
                # loss = loss_action + loss_done
                val_loss += loss.item() * feat_pairs.size(0)
                act_preds = torch.argmax(action_logits, dim=1)
                val_act_correct += (act_preds == action_labels).sum().item()
                done_preds = (torch.sigmoid(done_logits) > 0.5).float()
                val_done_correct += (done_preds == dones).sum().item()
                num_val_samples += feat_pairs.size(0)

        epoch_val_loss = val_loss / num_val_samples
        epoch_val_act_acc = val_act_correct / num_val_samples
        epoch_val_done_acc = val_done_correct / num_val_samples

        # 打印Epoch
        print(f"Epoch [{epoch + 1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Act Acc: {epoch_train_act_acc * 100:.2f}% | Done Acc: {epoch_train_done_acc * 100:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Act Acc: {epoch_val_act_acc * 100:.2f}% | Val Done Acc: {epoch_val_done_acc * 100:.2f}%")
        # train_losses.append(epoch_train_loss)
        # val_losses.append(epoch_val_loss)
        # train_act_acc.append(epoch_train_act_acc)
        # val_act_acc.append(epoch_val_act_acc)
        # train_done_acc.append(epoch_train_done_acc)
        # val_done_acc.append(epoch_val_done_acc)

        # 保存最佳模型
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "weights/new_nav_model.pth")
            # torch.save(model.state_dict(), "weights/real/best_nav_model.pth")
            print(f"刷新验证集表现，模型已保存")
    # print(train_losses)
    # print(val_losses)
    # print(train_act_acc)
    # print(val_act_acc)
    # print(train_done_acc)
    # print(val_done_acc)


if __name__ == "__main__":
    main()
