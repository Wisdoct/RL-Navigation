import os
import socket
import time
import struct
import numpy as np
import cv2
import torch
from transformers import ViTImageProcessor, ViTModel

from controller.model.controller import MLPController, AttentionControllerShortcut


def receive_all(conn, count):
    buf = b''
    while count:
        newbuf = conn.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


PORT = 9999
MODEL_PATH = "weights/real/best_nav_model.pth"
TARGET_IMG_PATH = "test_image/target3.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained("model/ViT16")
vit_model = ViTModel.from_pretrained("model/ViT16").to(device)
vit_model.eval()

# nav_model = MLPController(feat_dim=768, hidden_dim=1024, num_actions=3).to(device)
nav_model = AttentionControllerShortcut(feat_dim=768, hidden_dim=512, m_dim=16, num_actions=3).to(device)
# nav_model.load_state_dict(torch.load('./weights/best_nav_model.pth'))
nav_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
nav_model.eval()

target_bgr = cv2.imread(TARGET_IMG_PATH)
target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
with torch.no_grad():
    inputs = processor(images=target_rgb, return_tensors="pt").to(device)
    e_next = vit_model(**inputs).last_hidden_state[0, 0, :]  # [768]

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", PORT))   # 监听所有本地地址
server_socket.listen(1)
print(f"服务器已启动，等待客户端连接 (Port: {PORT})...")

conn, addr = server_socket.accept()
print(f"连接到客户端: {addr}")

# 数据包结构: [!f I] + 4字节超声波 + 4字节图像大小
header_struct = struct.Struct('!f I')

try:
    while True:
        # 接收数据包头
        header = receive_all(conn, header_struct.size)
        if not header:
            print("客户端断开连接。")
            break

        # 解包包头
        sonar_dist, img_size = header_struct.unpack(header)
        # 读取指定字长图像二进制字节流
        img_bytes = receive_all(conn, img_size)
        if not img_bytes:
            print("未读到完整的图像数据流，丢弃本帧。")
            break

        # 转化为cv2图像
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("图像解码失败。")
            continue
        # 转化为np图像
        current_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_rgb = cv2.resize(current_rgb, (256, 256))

        with torch.no_grad():
            inputs_t = processor(images=current_rgb, return_tensors="pt").to(device)
            e_t = vit_model(**inputs_t).last_hidden_state[0, 0, :]  # [768]
            feat_pair = torch.stack([e_t, e_next], dim=0).unsqueeze(0)  # [1, 2, 768]
            action_logits, done_logits = nav_model(feat_pair)
            action = torch.argmax(action_logits, dim=1).item()
            done_prob = torch.sigmoid(done_logits).item()
        print(f"Done Prob: {done_prob:.4f}")

        if done_prob > 0.5:
            cmd_str = "STOP\n"
            conn.sendall(cmd_str.encode('utf-8'))
            print("到达目的地，下发 STOP 信号！")
            break
        else:
            cmd_str = f"{action}\n"     # "0\n" ...
            conn.sendall(cmd_str.encode('utf-8'))

        cv2.imshow("Client Observation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(0.1)

except Exception as e:
    print(f"error: {e}")
finally:
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()
    print("控制端已安全断开并关闭。")
