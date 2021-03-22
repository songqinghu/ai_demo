import socket
import threading


def request(conn_socket, addr):
    print("客户端地址:", addr)
    # 接收连接信息
    recv_data = conn_socket.recv(1024)
    print("接收到的数据:", recv_data.decode())
    # 响应信息
    conn_socket.send("你的数据已被接收".encode())
    # 关闭当前连接
    conn_socket.close()


if __name__ == '__main__':
    # 创建套接字
    tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定地址
    tcp_server.bind(("", 8888))
    # 最大并发数
    tcp_server.listen(128)
    while True:
        # 获取当前连接
        conn_socket, addr = tcp_server.accept()
        client_thread = threading.Thread(target=request, args=(conn_socket, addr))
        client_thread.start()
    # 关闭服务器
    tcp_server.close()
