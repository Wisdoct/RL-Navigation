import socket
import struct
import cv2


def main():
    SERVER_IP = "192.168.1.118"
    PORT = 9999

    print("正在初始化摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("错误: 无法从摄像头获取画面！")
        return

    result, img_encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not result:
        print("错误: 图像压缩失败。")
        return

    img_bytes = img_encoded.tobytes()
    img_size = len(img_bytes)

    print(f"连接服务器 [{SERVER_IP}:{PORT}]...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    header_struct = struct.Struct("!I")

    try:
        client_socket.connect((SERVER_IP, PORT))
        print("成功连接到服务器！开始发送数据...")

        header = header_struct.pack(img_size)

        client_socket.sendall(header)
        client_socket.sendall(img_bytes)
        print(f"成功发送图片数据，大小: {img_size} 字节。")

    except Exception as e:
        print(f"发送出错: {e}")
    finally:
        client_socket.close()
        print("客户端已安全退出。")


if __name__ == "__main__":
    main()
