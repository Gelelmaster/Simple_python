import sys
import socket
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QTextEdit, QLineEdit, QPushButton

class ClientApp(QWidget):
    def __init__(self):
        super().__init__()

        self.client_socket = self.connect_to_server()

        self.setWindowTitle("角色选择与消息发送")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # 角色选择
        self.character_label = QLabel("选择角色：")
        self.character_combo = QComboBox()

        # 情感选择
        self.emotion_label = QLabel("选择情感：")
        self.emotion_combo = QComboBox()

        # 消息输入框
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("输入消息")

        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)

        # 聊天窗口
        self.chat_window = QTextEdit()
        self.chat_window.setReadOnly(True)

        # 布局
        layout.addWidget(self.character_label)
        layout.addWidget(self.character_combo)
        layout.addWidget(self.emotion_label)
        layout.addWidget(self.emotion_combo)
        layout.addWidget(self.message_input)
        layout.addWidget(self.send_button)
        layout.addWidget(self.chat_window)

        self.setLayout(layout)

        # 请求角色和情感列表
        self.request_character_and_emotion()

    def connect_to_server(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('127.0.0.1', 5555))  # 连接到本地的服务器
        return client_socket

    def request_character_and_emotion(self):
        """请求服务器返回角色和情感列表"""
        # 发送请求消息（可以自定义请求类型）
        self.client_socket.send("get_characters".encode('utf-8'))

        # 接收服务器响应
        data = self.client_socket.recv(1024).decode('utf-8')

        # 假设返回的是JSON格式的数据
        try:
            characters_and_emotions = json.loads(data)
            # 更新角色列表
            self.character_combo.addItems(characters_and_emotions.keys())
            # 默认选择第一个角色
            if characters_and_emotions:
                self.character_combo.setCurrentIndex(0)
                self.update_emotions(characters_and_emotions)

        except json.JSONDecodeError:
            print("Failed to decode the characters and emotions data.")
            return

    def update_emotions(self, characters_and_emotions):
        """更新情感下拉框"""
        selected_character = self.character_combo.currentText()
        emotions = characters_and_emotions.get(selected_character, [])
        self.emotion_combo.clear()
        self.emotion_combo.addItems(emotions)

    def send_message(self):
        message = self.message_input.text().strip()
        if message:
            character = self.character_combo.currentText()
            emotion = self.emotion_combo.currentText()

            # 显示自己发送的消息
            self.chat_window.append(f"我（{character}, {emotion}）: {message}")

            # 发送角色、情感和消息到服务端
            message_to_send = f"{character},{emotion},{message}"
            self.client_socket.send(message_to_send.encode('utf-8'))

            self.message_input.clear()

    def closeEvent(self, event):
        self.client_socket.send("close".encode('utf-8'))  # 发送关闭消息到服务端
        self.client_socket.close()  # 关闭socket连接
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = ClientApp()
    client.show()
    sys.exit(app.exec_())