import socket
import requests

def get_local_ip():
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e:
        return f"无法获取本地IP地址: {e}"

def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        public_ip = response.json()['ip']
        return public_ip
    except Exception as e:
        return f"无法获取公网IP地址: {e}"

if __name__ == "__main__":
    print("本地IP地址:", get_local_ip())
    print("公网IP地址:", get_public_ip())
    input("按任意键退出...")