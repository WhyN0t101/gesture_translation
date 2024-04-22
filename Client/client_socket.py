import socket 
class ClientSocket:
    def __init__(self, server_address='127.0.0.1', server_port=12345):
        self.server_address = server_address
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_address, self.server_port))

    def send(self, data):
        self.client_socket.sendall(data)

    def recv(self, bufsize):
        return self.client_socket.recv(bufsize)

    def close(self):
        self.client_socket.close()
