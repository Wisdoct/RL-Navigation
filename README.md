# RL-Navigation

An implement of navigation control using Reinforcement Learning and based on Mecanum car.
The contoller should be trained for two stages and can be tested in real-world:

## 1 Datasets collecting

Collect datasets in sim-world and real-world separately by:

### 1.1 in sim-world

Run **habitat_explorer/dagger_train.py** to collect datasets in Habitat simulator. Datasets will be saved at habitat_explorer/record-sim.

I used habitat-sim-0.2.5 and habitat-lab-0.2.5. Note that Habitat requires Linux operating system and cuda devices.

### 1.2 in real-world

Run **ppo_explorer/ppo_server.py** on PC and run **ppo_explorer/ppo_client.py** on mecanum car. **ppo_client** is based on lib functions pre-defined by robot manufacturer while **ppo_explorer** needs ViT encoder model which can be downloaded at [google/vit-base-patch16-224-in21k · Hugging Face](https://huggingface.co/google/vit-base-patch16-224-in21k).

## 2 Training

Move datasets(episode_*.pt) to controller/datasets/record_sim and controller/datasets/record_real for sim-world trajectories and real-world trajectories respectively.

Train controller model by running **controller/train.py**.

## 3 Navigate in real-world

Upload a test photo of a target location in real-world to controller/test_image. (You can also try using vehicle to take a photo by running controller/test_image/take_photo.) Then run **controller/run_server.py** and **controller/run_client.py** to test navigating in real-world.
