import sys
import time
import socket
import struct
import cv2
import numpy as np

# 导入麦克纳姆小车驱动与 Yahboom 传感器库
sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *
from Raspbot_Lib import Raspbot

bot = Raspbot()
bot.Ctrl_Servo(1, 100)
bot.Ctrl_Servo(2, 40)
bot.Ctrl_Ulatist_Switch(1)  # 开启超声波模块


def send_info(client_socket, cap, header_struct):
    ''' 采集超声波测距与摄像头画面，打包发送
    :param client_socket: socket对象
    :param cap: cv2摄像头对象
    :param header_struct: 数据头结构, struct对象
    :return: None
    '''
    # 超声波测距
    try:
        distance_mm = bot.read_data_array(0x1b, 1)[0]   # mm
        sonar_dist_m = float(distance_mm) / 1000.0
    except Exception:
        sonar_dist_m = 1.0

    # 读取摄像头
    ret, frame = cap.read()
    if not ret:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result, img_encode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_bytes = img_encode.tobytes()
    img_size = len(img_bytes)

    # 打包发送
    client_socket.sendall(header_struct.pack(sonar_dist_m, img_size))
    client_socket.sendall(img_bytes)


def main():
    SERVER_IP = "192.168.1.100"
    PORT = 9999
    speed = 10

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法开启车载摄像头。")
        return

    print(f"等待连接 [{SERVER_IP}:{PORT}]...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # IPv4 TCP

    try:
        client_socket.connect((SERVER_IP, PORT))
        print("握手成功")

        socket_file = client_socket.makefile('r', encoding='utf-8')
        header_struct = struct.Struct('!f I')

        while True:
            line = socket_file.readline()
            if not line:
                break

            cmd = line.strip()

            # 情况 A：收到环境重置指令
            if cmd == "RESET":
                print(">>> 收到环境 RESET 指令。")
                stop_robot()
                send_info(client_socket, cap, header_struct)
                continue

            # 情况 B：【核心修改】收到碰撞紧急刹车停止指令，触发物理后退逃逸
            elif cmd == "CRASH_STOP":
                print("🚨 收到 CRASH_STOP 信号！智能体撞击障碍物。执行紧急后退...")
                stop_robot()
                time.sleep(0.05)
                # 向后退 1 秒以脱离碰撞死锁区域
                move_backward(speed)
                time.sleep(1.0)
                stop_robot()
                print("退回安全区域，等待下一个 Episode 重置...")
                continue

            # 情况 C：执行强化学习下发的正常离散步动作
            elif cmd == "0":
                move_forward(speed)
            elif cmd == "1":
                rotate_left(speed)
            elif cmd == "2":
                rotate_right(speed)

            time.sleep(0.05)

            # 当前步执行完毕，继续打包下一帧的超声波与图像信息，回传给 Server 端计算 next_state
            send_info(client_socket, cap, header_struct)

    except Exception as e:
        print(f"车端运行异常: {e}")
    finally:
        stop_robot()
        bot.Ctrl_Ulatist_Switch(0)  # 随进程退出关闭超声波模块
        cap.release()
        client_socket.close()
        print("硬件通信安全关闭。")


if __name__ == "__main__":
    main()