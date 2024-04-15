# main_client.py
from client_interface import App

if __name__ == "__main__":
    app = App()
    print("Client connected to the server.")
    app.mainloop()
