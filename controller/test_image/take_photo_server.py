import socket
import struct
import cv2
import numpy as np


def receive_all(conn, count):
    buf = b""
    while count:
        newbuf = conn.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def main():
    PORT = 9999
    SAVE_PATH = "received_target.jpg"

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", PORT))
    server_socket.listen(1)
    print(f"服务器已启动，等待客户端连接 (Port: {PORT})...")

    conn, addr = server_socket.accept()
    print(f"连接到客户端: {addr}")

    header_struct = struct.Struct("!I")

    try:
        header = receive_all(conn, header_struct.size)
        if not header:
            print("未接收到有效的包头数据。")
            return

        (img_size,) = header_struct.unpack(header)
        print(f"读取到包头，即将接收的图片大小为: {img_size} 字节")

        img_bytes = receive_all(conn, img_size)
        if not img_bytes:
            print("未读取到完整的图像数据流。")
            return

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imwrite(SAVE_PATH, frame)
        else:
            print("错误: 图像解码失败。")

    except Exception as e:
        print(f"服务器出错: {e}")
    finally:
        conn.close()
        server_socket.close()
        print("服务器端已断开。")


if __name__ == "__main__":
    main()
