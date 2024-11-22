import sys
import socket
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ChatClient(QWidget):
    message_received = pyqtSignal(str)  # 自定义信号，用于显示收到的消息

    def __init__(self):
        super().__init__()
        self.init_ui()

        # 连接服务器
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('127.0.0.1', 5555))

        # 启动接收消息的线程
        self.receive_thread = threading.Thread(target=self.receive_message, daemon=True)
        self.receive_thread.start()

    def init_ui(self):
        self.setWindowTitle("微信样式聊天程序")
        self.setGeometry(100, 100, 400, 600)

        # 聊天显示窗口（带滚动功能）
        self.chat_area = QScrollArea(self)
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setStyleSheet("""
            background-color: #F5F5F5;
            border: none;  /* 隐藏边框 */
            QScrollBar:vertical { width: 0px; }  /* 隐藏垂直滚动条 */
            QScrollBar:horizontal { height: 0px; }  /* 隐藏水平滚动条 */
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
        self.message_entry.setFixedHeight(40)  # 初始高度为 40px

        # 发送按钮，增加按钮宽度
        self.send_button = QPushButton("发送", self)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12pt;
                border-radius: 10px;
                padding: 10px 20px;
                width: 40px;  /* 增加按钮宽度 */
            }
            QPushButton:pressed {
                background-color: #45A049;
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        # 布局
        entry_layout = QHBoxLayout()
        entry_layout.addWidget(self.message_entry)
        entry_layout.addWidget(self.send_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.chat_area)
        main_layout.addLayout(entry_layout)

        self.setLayout(main_layout)

        # 绑定消息接收信号
        self.message_received.connect(self.add_message_bubble)

    def resizeEvent(self, event):
        """窗口大小变化时调用，更新消息气泡的最大宽度"""
        window_width = self.width()
        self.max_bubble_width = int(window_width * 1.2)  # 设置气泡最大宽度为窗口宽度的40%
        super().resizeEvent(event)

    def add_message_bubble(self, message, position="left"):
        """
        显示消息气泡：左侧为接收的消息，右侧为发送的消息，并支持文字选择
        """
        bubble_color = "#E5E5E5" if position == "left" else "#4CAF50"
        text_color = "#000000" if position == "left" else "#FFFFFF"
        alignment = Qt.AlignLeft if position == "left" else Qt.AlignRight

        # 创建气泡框，并启用文字选择功能
        bubble = QLabel(message)
        bubble.setStyleSheet(
            f"background-color: {bubble_color}; color: {text_color}; font-size: 12pt; padding: 10px; "
            f"border-radius: 10px; max-width: {self.max_bubble_width}px;"
        )
        bubble.setWordWrap(True)
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 启用鼠标选择文字

        # 气泡布局
        bubble_layout = QHBoxLayout()
        if position == "left":
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()  # 左侧消息右对齐空白
        else:
            bubble_layout.addStretch()  # 右侧消息左对齐空白
            bubble_layout.addWidget(bubble)

        # 外层容器用于存储气泡并对齐
        bubble_container = QFrame()
        bubble_container.setLayout(bubble_layout)

        self.chat_content.addWidget(bubble_container)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())


    def send_message(self):
        message = self.message_entry.text().strip()
        if message:
            self.add_message_bubble(message, "right")  # 显示发送的消息
            self.client_socket.send(message.encode('utf-8'))  # 发送消息到服务器
            self.message_entry.clear()

    def receive_message(self):
        while True:
            try:
                message = self.client_socket.recv(1024).decode('utf-8')
                if message:
                    self.message_received.emit(message)  # 触发显示消息的信号
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

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 12)  # 设置字体为微软雅黑，大小为12
    app.setFont(font)

    client = ChatClient()
    client.show()
    sys.exit(app.exec_())
