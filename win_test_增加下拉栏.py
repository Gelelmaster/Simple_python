import os
import sys
import socket
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ChatClient(QWidget):
    message_received = pyqtSignal(str)  # 自定义信号，用于显示收到的消息

    def __init__(self):
        super().__init__()
        self.init_ui()

        # 检查或创建 trained 文件夹
        self.init_trained_folder()

        # 连接服务器
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('127.0.0.1', 5555))

        # 启动接收消息的线程
        self.receive_thread = threading.Thread(target=self.receive_message, daemon=True)
        self.receive_thread.start()

    def init_trained_folder(self):
        """检查并创建 trained 文件夹"""
        self.trained_folder = "trained"
        if not os.path.exists(self.trained_folder):
            os.makedirs(self.trained_folder)
        self.update_file_list()

    def update_file_list(self):
        """更新下拉菜单中的文件列表"""
        files = os.listdir(self.trained_folder)
        self.file_selector.clear()
        if files:
            self.file_selector.addItems(files)
        else:
            self.file_selector.addItem("（无文件）")

    def toggle_dropdown(self):
        """切换下拉菜单显示/隐藏"""
        is_visible = self.file_selector.isVisible()
        self.file_selector.setVisible(not is_visible)

    def init_ui(self):
        self.setWindowTitle("微信样式聊天程序")
        self.setGeometry(100, 100, 400, 600)

        # 聊天显示窗口（带滚动功能）
        self.chat_area = QScrollArea(self)
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setStyleSheet("""
            background-color: #F5F5F5;
            border: none;
            QScrollBar:vertical { width: 0px; }
            QScrollBar:horizontal { height: 0px; }
        """)

        # 聊天内容容器
        self.chat_content = QVBoxLayout()
        self.chat_content.setAlignment(Qt.AlignTop)

        # 容器的父控件
        content_widget = QFrame()
        content_widget.setLayout(self.chat_content)
        self.chat_area.setWidget(content_widget)

        # 消息输入框
        self.message_entry = QLineEdit(self)
        self.message_entry.setPlaceholderText("输入消息...")
        self.message_entry.setStyleSheet("""
            font-size: 12pt;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        """)
        self.message_entry.setFixedHeight(40)

        # 发送按钮
        self.send_button = QPushButton("发送", self)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12pt;
                border-radius: 10px;
                padding: 10px 20px;
                width: 40px;
            }
            QPushButton:pressed {
                background-color: #45A049;
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        # 文件选择下拉菜单（默认隐藏）
        self.file_selector = QComboBox(self)
        self.file_selector.setVisible(False)  # 初始状态隐藏
        self.file_selector.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)

        # 切换按钮
        self.toggle_button = QPushButton("显示/隐藏文件列表", self)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
                background-color: #F5F5F5;
            }
            QPushButton:pressed {
                background-color: #E0E0E0;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_dropdown)

        # 布局
        entry_layout = QHBoxLayout()
        entry_layout.addWidget(self.message_entry)
        entry_layout.addWidget(self.send_button)

        menu_layout = QVBoxLayout()
        menu_layout.addWidget(self.toggle_button)
        menu_layout.addWidget(self.file_selector)

        main_layout = QVBoxLayout()
        main_layout.addLayout(menu_layout)
        main_layout.addWidget(self.chat_area)
        main_layout.addLayout(entry_layout)

        self.setLayout(main_layout)

        # 绑定消息接收信号
        self.message_received.connect(self.add_message_bubble)

    def resizeEvent(self, event):
        window_width = self.width()
        self.max_bubble_width = int(window_width * 1.2)
        super().resizeEvent(event)

    def add_message_bubble(self, message, position="left"):
        bubble_color = "#E5E5E5" if position == "left" else "#4CAF50"
        text_color = "#000000" if position == "left" else "#FFFFFF"

        bubble = QLabel(message)
        bubble.setStyleSheet(
            f"background-color: {bubble_color}; color: {text_color}; font-size: 12pt; padding: 10px; "
            f"border-radius: 10px; max-width: {self.max_bubble_width}px;"
        )
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)

        bubble_layout = QHBoxLayout()
        if position == "left":
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()
        else:
            bubble_layout.addStretch()
            bubble_layout.addWidget(bubble)

        bubble_container = QFrame()
        bubble_container.setLayout(bubble_layout)

        self.chat_content.addWidget(bubble_container)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def send_message(self):
        message = self.message_entry.text().strip()
        if message:
            self.add_message_bubble(message, "right")
            self.client_socket.send(message.encode('utf-8'))
            self.message_entry.clear()

    def receive_message(self):
        while True:
            try:
                message = self.client_socket.recv(1024).decode('utf-8')
                if message:
                    self.message_received.emit(message)
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

    def closeEvent(self, event):
        try:
            self.client_socket.send("close".encode('utf-8'))
        except Exception as e:
            print(f"Error sending close message: {e}")
        self.client_socket.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    font = QFont("Microsoft YaHei", 12)
    app.setFont(font)

    client = ChatClient()
    client.show()
    sys.exit(app.exec_())

'''
下拉栏相关代码

import os
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QComboBox

class ChatClient(QWidget):
    def init_train_folder(self):
        """检查并创建 train 文件夹"""
        self.train_folder = "train"
        if not os.path.exists(self.train_folder):
            os.makedirs(self.train_folder)
        self.update_file_list()

    def update_file_list(self):
        """更新下拉菜单中的文件列表"""
        files = os.listdir(self.train_folder)
        self.file_selector.clear()
        if files:
            self.file_selector.addItems(files)
        else:
            self.file_selector.addItem("（无文件）")

    def toggle_dropdown(self):
        """切换下拉菜单显示/隐藏"""
        is_visible = self.file_selector.isVisible()
        self.file_selector.setVisible(not is_visible)

    def init_ui(self):
        # 文件选择下拉菜单（默认隐藏）
        self.file_selector = QComboBox(self)
        self.file_selector.setVisible(False)  # 初始状态隐藏
        self.file_selector.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)

        # 切换按钮
        self.toggle_button = QPushButton("显示/隐藏文件列表", self)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
                background-color: #F5F5F5;
            }
            QPushButton:pressed {
                background-color: #E0E0E0;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_dropdown)

        # 布局（仅与下拉栏相关的部分）
        menu_layout = QVBoxLayout()
        menu_layout.addWidget(self.toggle_button)
        menu_layout.addWidget(self.file_selector)

        # 将 `menu_layout` 添加到主界面布局中
        self.setLayout(menu_layout)
'''