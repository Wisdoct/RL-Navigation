#!/usr/bin/env python3

import os
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTModel

import habitat
import habitat_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower  # 引入专家跟随器
from habitat.utils.visualizations import maps  # 引入地图可视化工具
from habitat.utils.visualizations.utils import images_to_video  # 引入视频保存工具
from PPO import PPO


TOTAL_EPISODES = 71
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 3e-5
LR_CRITIC = 1e-4
GAMMA = 0.99
LAMBDA = 0.95
EPOCHS = 10
EPS = 0.2
HIDDEN_DIM = 1024
MAX_STEPS = 200
# 惩罚距离设定
OBSTACLE_THRESHOLD = 0.1
MIN_CRITICAL_DIST = 0.01
# dagger比例
EXPERT_RATIO = 0.9

DATA_OUT_DIR = os.path.join("collected_data")
VIDEO_OUT_DIR = os.path.join("collected_videos")


class ExplorationRLEnv(habitat.RLEnv):
    def __init__(self, config):
        super().__init__(config)
        self.step_count = 0
        self.start_position = None
        self.goal_position = None
        self.max_dist = 0.0
        self.last_dist_to_obstacle = None

    def get_reward_range(self):
        return [-10.0, 1.0]

    def get_reward(self, observations):
        return 0.0

    def get_done(self, observations):
        if self.step_count >= MAX_STEPS: return True
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def reset_state(self):
        self.step_count = 0
        self.max_dist = 0.0
        self.reset()

        # 1. 随机抽取起点
        random_position = self.habitat_env.sim.sample_navigable_point()
        random_yaw = np.random.uniform(0, 2 * np.pi)
        random_rotation = [0.0, np.sin(random_yaw / 2.0), 0.0, np.cos(random_yaw / 2.0)]

        # 2. 实例化寻路组件，用于计算测地线距离
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(random_position)

        # 3. 循环抽取目标点，直到与起点的地理距离（测地线距离） >= 1.0 米
        while True:
            sampled_goal = self.habitat_env.sim.sample_navigable_point()
            path.requested_end = np.array(sampled_goal)

            # 进行路径规划验证
            if self.habitat_env.sim.pathfinder.find_path(path):
                geodesic_dist = path.geodesic_distance
                # 排除无穷大（不可达）且确保距离不小于 1 米
                if not np.isinf(geodesic_dist) and geodesic_dist >= 1.0:
                    self.goal_position = sampled_goal
                    break
            # 如果不满足条件或寻路失败，则在下一轮循环中重新抽样目标点

        # 4. 设定智能体状态
        agent_state = self.habitat_env.sim.get_agent_state()
        agent_state.position = random_position
        agent_state.rotation = random_rotation
        self.habitat_env.sim.set_agent_state(agent_state.position, agent_state.rotation)

        self.start_position = np.array(random_position)

        init_dist = self.habitat_env.sim.pathfinder.distance_to_closest_obstacle(
            self.start_position, max_search_radius=2.0
        )
        self.last_dist_to_obstacle = 0.0 if np.isnan(init_dist) or init_dist < 0 else init_dist

        return self.habitat_env.sim.get_observations_at(position=random_position, rotation=random_rotation)


class ViTEncoder:
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained("ViT16")
        self.model = ViTModel.from_pretrained("ViT16").to(DEVICE)
        self.model.eval()

    def extract(self, rgb_image):
        with torch.no_grad():
            inputs = self.processor(images=rgb_image, return_tensors="pt").to(DEVICE)
            outputs = self.model(**inputs)
            cls_feature = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        return cls_feature


if __name__ == "__main__":
    config = habitat.get_config(
        config_path="../benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"    # 绘图组件
        ],
    )

    encoder = ViTEncoder()
    agent = PPO(
        state_dim=768, hidden_dim=HIDDEN_DIM, action_dim=3,
        actor_lr=LR_ACTOR, critic_lr=LR_CRITIC, lmbda=LAMBDA,
        epochs=EPOCHS, eps=EPS, gamma=GAMMA, device=DEVICE
    )

    num_episodes = 10
    with ExplorationRLEnv(config=config) as env:
        # 初始化follower
        forward_step_size = config.habitat.simulator.forward_step_size
        follower = ShortestPathFollower(env.habitat_env.sim, forward_step_size, False)
        for episode in range(num_episodes):
            obs = env.reset_state()
            state = encoder.extract(obs["rgb"])
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
            done = False
            episode_reward = 0
            video_images = []  # 保存最后一集

            while not done:
                env.step_count += 1
                expert_action = follower.get_next_action(env.goal_position)
                if expert_action is None or expert_action < 1:
                    EXPERT_RATIO = 0    # 已经到达follower的目标，之后仅选择智能体动作
                expert_action = expert_action - 1   # [1,2,3] -> [0,1,2]
                agent_action = agent.take_action(state)
                if np.random.rand() < EXPERT_RATIO:
                    action = expert_action
                else:
                    action = agent_action
                habitat_action = action + 1

                next_obs, _, _, info = env.step(habitat_action)
                next_state = encoder.extract(next_obs["rgb"])

                # 最后一集保存观测与俯视图
                if episode == num_episodes - 1:
                    rgb_frame = next_obs["rgb"][:, :, :3]
                    if "top_down_map" in info:
                        top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                            info["top_down_map"], rgb_frame.shape[0]
                        )
                        # 将智能体视角与二维俯视图水平拼接在一起
                        combined_frame = np.concatenate((rgb_frame, top_down_map), axis=1)
                        video_images.append(combined_frame)
                    else:
                        video_images.append(rgb_frame)

                reward = 0.0
                done = False

                # 探索奖励
                current_pos = np.array(env.habitat_env.sim.get_agent_state().position)

                path_track = habitat_sim.ShortestPath()
                path_track.requested_start = env.start_position
                path_track.requested_end = current_pos

                if env.habitat_env.sim.pathfinder.find_path(path_track):
                    dist_from_start = path_track.geodesic_distance
                    if np.isinf(dist_from_start):
                        dist_from_start = 0.0
                else:
                    dist_from_start = 0.0

                if dist_from_start > env.max_dist:
                    reward += 1.0
                    env.max_dist = dist_from_start

                # 障碍物碰撞负奖励
                dist_to_obstacle = env.habitat_env.sim.pathfinder.distance_to_closest_obstacle(
                    current_pos, max_search_radius=2.0
                )
                if np.isnan(dist_to_obstacle) or dist_to_obstacle < 0:
                    dist_to_obstacle = 0.0
                if dist_to_obstacle <= MIN_CRITICAL_DIST:
                    reward = -5.0
                    done = True
                # if dist_to_obstacle < OBSTACLE_THRESHOLD:
                #     if dist_to_obstacle < env.last_dist_to_obstacle:
                #         ratio = (dist_to_obstacle - MIN_CRITICAL_DIST) / (OBSTACLE_THRESHOLD - MIN_CRITICAL_DIST)
                #         ratio = max(0.0, min(1.0, ratio))
                #         reward += (-10.0 + 9.0 * ratio)
                #     if dist_to_obstacle <= MIN_CRITICAL_DIST:
                #         done = True
                # env.last_dist_to_obstacle = dist_to_obstacle

                if env.step_count >= MAX_STEPS or env.habitat_env.episode_over:
                    done = True

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_reward += reward

            # 超过15个动作的集可以被保存为数据集
            if len(transition_dict['states']) >= 15:
                save_data = {
                    "embeddings": torch.tensor(np.array(transition_dict['states']), dtype=torch.float32),
                    "actions": torch.tensor(transition_dict['actions'], dtype=torch.long)
                }
                save_path = os.path.join(DATA_OUT_DIR, f"episode_{TOTAL_EPISODES}.pt")
                torch.save(save_data, save_path)
                print(f"第 {TOTAL_EPISODES} 集数据已保存")
                TOTAL_EPISODES += 1

            # 最后一集保存视频
            if episode == num_episodes - 1 and len(video_images) > 0:
                video_name = f"trajectory"
                images_to_video(video_images, VIDEO_OUT_DIR, video_name)
                print(f"轨迹视频已保存")

            # 更新 PPO 策略
            agent.update(transition_dict)

            print(f"Episode: {episode + 1}/{num_episodes} | Reward: {episode_reward:.2f} | Max Dist: {env.max_dist:.2f} | Steps: {env.step_count}")

        os.makedirs("models", exist_ok=True)
        agent.save("models/actor.pth", "models/critic.pth")
