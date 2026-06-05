import os
import socket
import struct
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
from transformers import ViTImageProcessor, ViTModel
import gymnasium as gym
from gymnasium import spaces

from policy.PPO import PPO

EPISODE_N = 11


class RealCarTCPEnv(gym.Env):
    def __init__(self, host="0.0.0.0", port=9999, t_min=0.5, t_max=0.85, device=torch.device("cpu")):
        super(RealCarTCPEnv, self).__init__()
        self.device = device
        self.t_min = t_min  # 最大相似度 < t_min => 探索到新环境
        self.t_max = t_max  # 最大相似度 > t_max => 在已探索区域内
        self.waypoints = []
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32)
        self.processor = ViTImageProcessor.from_pretrained("ViT16")
        self.vit_model = ViTModel.from_pretrained("ViT16").to(self.device)
        self.vit_model.eval()

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"服务端已启动(Port: {port})，等待连接...")

        self.conn, addr = self.server_socket.accept()
        print(f"客户端已连接: {addr}")

        # 数据包结构: [!f I] + 4字节超声波 + 4字节图像大小
        self.header_struct = struct.Struct('!f I')

    def receive_all(self, count):
        buf = b''
        while count:
            newbuf = self.conn.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def get_observation(self):
        header = self.receive_all(self.header_struct.size)
        if not header:
            raise ConnectionResetError("error: 断开连接")
        sonar_dist, img_size = self.header_struct.unpack(header)
        img_bytes = self.receive_all(img_size)
        if not img_bytes:
            raise ConnectionResetError("error: 未读到完整图像数据流")

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        current_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.resize(current_rgb, (256, 256))
        cv2.imshow('Client Observation', frame)
        # cv2.waitKey(1)

        with torch.no_grad():
            inputs = self.processor(images=current_rgb, return_tensors="pt").to(self.device)
            outputs = self.vit_model(**inputs)
            state_feat = outputs.last_hidden_state[0, 0, :].cpu().numpy()   # [768]
        return state_feat, sonar_dist

    def reset(self, seed=None, options=None):
        print("-------- 环境重置 --------")
        self.conn.sendall("RESET\n".encode('utf-8'))
        state_feat, _ = self.get_observation()
        if len(self.waypoints) == 0:
            self.waypoints.append(state_feat)
        return state_feat, {}

    def step(self, action):
        ''' 步进
        :param action: 采取动作
        :return: next_state(obs[768]), reward, terminated, truncated, info
        '''
        cmd_str = f"{action}\n"
        self.conn.sendall(cmd_str.encode('utf-8'))
        next_state, sonar_dist = self.get_observation()
        reward = 0
        # 碰撞判定
        if sonar_dist < 0.2:
            print(f"[碰撞风险] 前方障碍物距离: {sonar_dist:.3f}m")
            self.conn.sendall("CRASH_STOP\n".encode('utf-8'))
            reward -= 3.0
            done = True
            return next_state, reward, done, False, {"is_crash": True}

        t_next_state = torch.tensor(next_state, dtype=torch.float32)
        waypoints = torch.tensor(np.array(self.waypoints), dtype=torch.float32)     # 先np.array后torch.tensor效率高
        # 计算最大相似度
        cos_sim = torch.nn.functional.cosine_similarity(t_next_state.unsqueeze(0), waypoints, dim=1)
        cos_sim_abs = torch.abs(cos_sim).numpy()
        max_sim = np.max(cos_sim_abs)

        # 最大相似度 < t_min => 探索到新环境
        if max_sim < self.t_min:
            reward += 1.0
            self.waypoints.append(next_state)
            print(
                f"[新区域发现] 最大相似度 {max_sim:.3f} < {self.t_min}, 路标总数: {len(self.waypoints)})")
        # 最大相似度 > t_max => 在已探索区域内
        elif max_sim > self.t_max:
            reward += -0.5
            print(f"[重复探索区域] 最大相似度 {max_sim:.3f} > {self.t_max}")
        if action == 0:     # 鼓励前进
            reward += 0.1
        done = False
        return next_state, reward, done, False, {"is_crash": False}

    def close(self):
        self.conn.close()
        self.server_socket.close()


def train_ppo():
    trajectory_dir = "./record_real"
    state_dim = 768
    action_dim = 3
    hidden_dim = 512
    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    num_episodes = 100
    truncated_steps = 50
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = RealCarTCPEnv(device=device, t_min=0.4, t_max=0.75)
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    agent.actor.load_state_dict(torch.load('./ppo_weights/ppo_actor.pth'))
    agent.critic.load_state_dict(torch.load('./ppo_weights/ppo_critic.pth'))

    global_steps = 0
    trajectory_memory = {"embeddings": [], "actions": []}

    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False

            transition_dict = {'states': [], 'actions': [], 'next_states':[], 'rewards': [], 'dones':[]}
            step_count = 0

            print(f"\n>> 开始 Episode {episode + 1}/{num_episodes}")
            while not done and step_count < truncated_steps:
                action = agent.take_action(state)
                next_state, reward, done, _, info = env.step(action)

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state = next_state
                step_count += 1

            print(f"本集探索 {step_count} 步。更新PPO。")
            agent.update(transition_dict)
            agent.save(f"./ppo_weights/ppo_actor_new.pth", f"./ppo_weights/ppo_critic_new.pth")

            # 若轨迹步数小于15步，不计入数据集
            if step_count < 20:
                continue

            for s, a in zip(transition_dict["states"], transition_dict["actions"]):
                trajectory_memory["embeddings"].append(s)
                trajectory_memory["actions"].append(a)
                global_steps += 1
            save_dataset = {
                "embeddings": torch.tensor(np.array(trajectory_memory["embeddings"]), dtype=torch.float32),
                "actions": torch.tensor(np.array(trajectory_memory["actions"]), dtype=torch.long)
            }
            pt_path = os.path.join(trajectory_dir, f"episode_{EPISODE_N}_step_{global_steps}.pt")
            torch.save(save_dataset, pt_path)
            print(f"[数据集保存成功] {step_count}步轨迹已写入: {pt_path}")
            trajectory_memory = {"embeddings": [], "actions": []}   # 清空轨迹数据集

    except Exception as e:
        print(f"训练异常中断: {e}")
    finally:
        env.close()


if __name__ == '__main__':
    train_ppo()
