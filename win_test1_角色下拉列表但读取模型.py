import sys
import socket
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import Qt
from Gptsovit_tts import get_characters_and_emotions  # 从Gptsovit_tts模块导入

# 连接到服务器的函数
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 5555))  # 连接到本地的服务器
    return client_socket

class ChatClient(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.client_socket = connect_to_server()

        # 启动接收消息的线程
        self.receive_thread = threading.Thread(target=self.receive_message, daemon=True)
        self.receive_thread.start()

    def init_ui(self):
        self.setWindowTitle("微信样式聊天程序")

        # 主布局
        layout = QVBoxLayout()

        # 初始化角色和情感字典
        self.characters_and_emotions_dict = get_characters_and_emotions()
        
        # 角色选择下拉列表
        self.character_label = QLabel("选择角色：")
        self.character_combo = QComboBox()
        self.character_combo.addItems(list(self.characters_and_emotions_dict.keys()))  # 添加角色列表
        self.character_combo.currentTextChanged.connect(self.update_emotions)  # 角色选择变化时更新情感选项
        layout.addWidget(self.character_label)
        layout.addWidget(self.character_combo)

        # 情感选择下拉列表
        self.emotion_label = QLabel("选择情感：")
        self.emotion_combo = QComboBox()
        self.update_emotions()  # 初始化情感选项
        layout.addWidget(self.emotion_label)
        layout.addWidget(self.emotion_combo)

        # 聊天显示窗口
        self.chat_window = QTextEdit()
        self.chat_window.setReadOnly(True)
        layout.addWidget(self.chat_window)

        # 消息输入框
        self.message_entry = QLineEdit()
        self.message_entry.setPlaceholderText("输入消息...")
        layout.addWidget(self.message_entry)

        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)

        self.setLayout(layout)

    # 更新情感下拉菜单选项
    def update_emotions(self):
        character = self.character_combo.currentText()
        emotions = self.characters_and_emotions_dict.get(character, ["default"])
        self.emotion_combo.clear()
        self.emotion_combo.addItems(emotions)

    # 发送消息的函数
    def send_message(self):
        message = self.message_entry.text()
        if message.strip():  # 确保消息不是空白
            # 获取用户选择的角色和情感
            character = self.character_combo.currentText()
            emotion = self.emotion_combo.currentText()
            
            # 显示发送的消息并清空输入框
            self.chat_window.append(f"我（{character}, {emotion}）: {message}")
            
            # 发送角色、情感和消息内容到服务端
            message_to_send = f"{character},{emotion},{message}"
            self.client_socket.send(message_to_send.encode('utf-8'))
            self.message_entry.clear()

    # 接收消息的函数
    def receive_message(self):
        while True:
            try:
                message = self.client_socket.recv(1024).decode('utf-8')
                if message:
                    # 显示接收到的消息
                    self.chat_window.append(f"AI: {message}")
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

    # 客户端关闭时通知服务端
    def closeEvent(self, event):
        self.client_socket.send("close".encode('utf-8'))  # 发送关闭消息到服务端
        self.client_socket.close()  # 关闭socket连接
        event.accept()  # 关闭窗口

# 启动应用程序
app = QApplication(sys.argv)
client = ChatClient()
client.resize(400, 600)
client.show()
sys.exit(app.exec_())
