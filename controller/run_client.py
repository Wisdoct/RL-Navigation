import sys
import time
import socket
import struct
import cv2  # 引入 OpenCV 捕获车载图像

# 保持你原有的底盘及外设硬件导入与初始化
sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *
from Raspbot_Lib import Raspbot

bot = Raspbot()
bot.Ctrl_Servo(1, 100)
bot.Ctrl_Servo(2, 40)
bot.Ctrl_Ulatist_Switch(1)  # 开启超声波


def main():
    # --- 远程控制服务器配置 ---
    SERVER_IP = "192.168.1.118"  # ⚠️ 请修改为你电脑在局域网内的真实 IP
    PORT = 9999
    speed = 10  # 保持你原有的设定速度

    print("正在初始化车载摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 无法打开车载摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    print(f"正在尝试连接本机控制端服务器 [{SERVER_IP}:{PORT}]...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    header_struct = struct.Struct('!f I')

    try:
        client_socket.connect((SERVER_IP, PORT))
        print("成功连接到控制端！开始执行同步交互流程...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("警告: 无法从摄像头获取画面，跳过本帧。")
                time.sleep(0.05)
                continue

            try:
                sonar_dist = float(bot.Get_Ulatist())
            except Exception:
                sonar_dist = 9.9

            result, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not result:
                print("图像压缩失败。")
                continue

            img_bytes = img_encoded.tobytes()
            img_size = len(img_bytes)

            header = header_struct.pack(sonar_dist, img_size)

            client_socket.sendall(header)
            client_socket.sendall(img_bytes)

            data = b''
            while b'\n' not in data:
                chunk = client_socket.recv(1)
                if not chunk:
                    raise ConnectionResetError("服务器意外断开连接")
                data += chunk

            cmd = data.decode('utf-8').strip()
            print(f"发送数据成功 -> 收到服务器回传指令: {cmd}")

            if cmd == "STOP":
                print("收到停止指令，测试结束。")
                stop_robot()
                break

            elif cmd == "0":
                print("动作: 前进")
                move_forward(speed)

            elif cmd == "1":
                print("动作: 左转")
                rotate_left(speed)

            elif cmd == "2":
                print("动作: 右转")
                rotate_right(speed)

            else:
                print(f"收到未知指令: {cmd}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n用户手动中断，正在紧急刹车...")
    except Exception as e:
        print(f"小车运行出错: {e}")
    finally:
        stop_robot()  # 确保安全静止
        cap.release()  # 释放摄像头
        client_socket.close()
        print("小车客户端已安全退出。")


if __name__ == "__main__":
    main()
