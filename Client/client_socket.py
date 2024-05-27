import socket

class ClientSocket:
    """A class for handling client-side socket connections."""

    def __init__(self, server_address='127.0.0.1', server_port=12345):
        """Initialize the client socket with the given server address and port."""
        self.server_address = server_address
        self.server_port = server_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_address, self.server_port))

    def send(self, data):
        """Send data to the server."""
        self.client_socket.sendall(data)

    def recv(self, bufsize):
        """Receive data from the server."""
        return self.client_socket.recv(bufsize)

    def close(self):
        """Close the client socket."""
        self.client_socket.close()

    def is_socket_open(self):
        """Check if the client socket is open."""
        return not self.client_socket._closed
