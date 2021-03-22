import socket

"""
TCP客户端demo
"""
if __name__ == '__main__':
    # 创建套接字对象
    tcp_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 和服务器连接
    tcp_client.connect(("127.0.0.1", 8888))
    # 发送数据
    tcp_client.send("hello world!".encode(encoding="utf-8"))
    # 接收数据
    recv_data = tcp_client.recv(1024)
    print(recv_data.decode())
    # 关闭连接
    tcp_client.close()
