import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionControllerShortcut(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256, m_dim=5, num_actions=3):
        '''
        :param feat_dim: ViT提取的特征为768维
        :param hidden_dim: 所有隐层维度(方便调参)
        :param m_dim: 键值对数
        :param num_actions: 动作维度
        '''
        super(AttentionControllerShortcut, self).__init__()

        self.hidden_dim = hidden_dim
        self.m_dim = m_dim

        self.q_proj = nn.Linear(feat_dim, hidden_dim)   # 仅1个查询
        self.k_proj = nn.Linear(feat_dim, m_dim * hidden_dim)
        self.v_proj = nn.Linear(feat_dim, m_dim * hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.self_attention_block = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.shortcut_dim = 256
        self.e_next_shortcut = nn.Sequential(
            nn.Linear(feat_dim, self.shortcut_dim),
            nn.LayerNorm(self.shortcut_dim),
            nn.ReLU()
        )

        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.done_head = nn.Linear(hidden_dim + self.shortcut_dim, 1)

    def forward(self, feat_pair):
        '''
        :param feat_pair: (batch, 2, hidden_dim)
        :return: 动作预测(batch, num_actions), 停止点预测(batch, 1)
        '''
        batch_size = feat_pair.size(0)

        e_curr = feat_pair[:, 0, :]
        e_goal = feat_pair[:, 1, :]

        Q = self.q_proj(e_goal).unsqueeze(1)
        K = self.k_proj(e_curr).view(batch_size, self.m_dim, self.hidden_dim)  # (batch, m_dim, hidden_dim)
        V = self.v_proj(e_curr).view(batch_size, self.m_dim, self.hidden_dim)  # (batch, m_dim, hidden_dim)
        scaling = self.hidden_dim ** 0.5
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scaling
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, m_dim)
        cross_feat = torch.bmm(attn_weights, V)

        refined_feat = self.self_attention_block(cross_feat).squeeze(1)
        # refined_feat = self.self_attention_block(h_feat)

        action_logits = self.action_head(refined_feat)
        e_next_feat = self.e_next_shortcut(e_goal)  # (batch, shortcut_dim)
        done_combined_feat = torch.cat([refined_feat, e_next_feat], dim=-1)  # (batch, hidden_dim + shortcut_dim)
        done_logits = self.done_head(done_combined_feat)  # (batch, 1)
        return action_logits, done_logits


class MLPController(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256, num_actions=3):
        super(MLPController, self).__init__()
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(self, feat_pair):
        batch_size = feat_pair.size(0)
        x = feat_pair.view(batch_size, -1)
        x = self.mlp(x)
        action_logits = self.action_head(x)
        done_logits = self.done_head(x)
        return action_logits, done_logits


class AttentionController(nn.Module):
    def __init__(self, feat_dim=768, hidden_dim=256, m_dim=5, num_actions=3):
        '''
        :param feat_dim: ViT提取的特征为768维 - int
        :param hidden_dim: 所有隐层维度(方便调参) - int
        :param m_dim: 键值对数 - int
        :param num_actions: 动作维度 - int
        '''
        super(AttentionController, self).__init__()

        self.hidden_dim = hidden_dim
        self.m_dim = m_dim

        self.q_proj = nn.Linear(feat_dim, hidden_dim)   # 仅1个查询
        self.k_proj = nn.Linear(feat_dim, m_dim * hidden_dim)
        self.v_proj = nn.Linear(feat_dim, m_dim * hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.self_attention_block = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.shortcut_dim = 256
        self.e_next_shortcut = nn.Sequential(
            nn.Linear(feat_dim, self.shortcut_dim),
            nn.LayerNorm(self.shortcut_dim),
            nn.ReLU()
        )

        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_pair):
        batch_size = feat_pair.size(0)
        e_t = feat_pair[:, 0, :]
        e_next = feat_pair[:, 1, :]
        Q = self.q_proj(e_next).unsqueeze(1)
        K = self.k_proj(e_t).view(batch_size, self.m_dim, self.hidden_dim)  # (batch, m_dim, hidden_dim)
        V = self.v_proj(e_t).view(batch_size, self.m_dim, self.hidden_dim)  # (batch, m_dim, hidden_dim)
        scaling = self.hidden_dim ** 0.5
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scaling
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, m_dim)
        cross_feat = torch.bmm(attn_weights, V)
        refined_feat = self.self_attention_block(cross_feat).squeeze(1)
        action_logits = self.action_head(refined_feat)
        done_logits = self.done_head(refined_feat)
        return action_logits, done_logits



if __name__ == "__main__":
    model = AttentionControllerShortcut(feat_dim=768, hidden_dim=256, num_actions=3)

    # 模拟从你的 DataLoader 出来的 batch 数据
    mock_batch_feat = torch.randn(32, 2, 768)  # batch=32

    act_out, done_out = model(mock_batch_feat)

    print(f"输入形状: {mock_batch_feat.shape}")
    print(f"动作预测输出形状 (应为 batch, 3): {act_out.shape}")
    print(f"停止信号输出形状 (应为 batch, 1): {done_out.shape}")
