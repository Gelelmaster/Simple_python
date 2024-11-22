import sys
import socket
import threading
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


class ChatClient(QWidget):
    message_received = pyqtSignal(str)  # 定义一个信号，用于在界面上显示收到的消息

    def __init__(self):
        super().__init__()
        self.init_ui()  # 初始化用户界面

        # 连接到服务器
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('127.0.0.1', 5555))  # 假设服务器运行在本地的 127.0.0.1:5555

        # 请求角色和情感数据
        self.request_character_and_emotion()

        # 启动线程以接收服务器的消息
        self.receive_thread = threading.Thread(target=self.receive_message, daemon=True)
        self.receive_thread.start()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("微信样式聊天程序")  # 设置窗口标题
        self.setGeometry(100, 100, 400, 600)  # 设置窗口大小和位置

        # 聊天显示区域
        self.chat_area = QScrollArea(self)
        self.chat_area.setWidgetResizable(True)  # 内容自适应大小
        self.chat_area.setStyleSheet("""
            background-color: #F5F5F5;
            border: none;
            QScrollBar:vertical { width: 0px; }
            QScrollBar:horizontal { height: 0px; }
        """)

        # 聊天内容容器
        self.chat_content = QVBoxLayout()
        self.chat_content.setAlignment(Qt.AlignTop)  # 内容从顶部开始排列
        content_widget = QFrame()  # 容器的父控件
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
        self.send_button.clicked.connect(self.send_message)  # 绑定点击事件

        # 初始化角色和情感选择
        self.character_selector = QComboBox(self)  # 角色选择框
        self.character_selector.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        self.emotion_selector = QComboBox(self)  # 情感选择框
        self.emotion_selector.setStyleSheet("""
            QComboBox {
                font-size: 12pt;
                padding: 5px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)

        # 布局
        entry_layout = QHBoxLayout()  # 消息输入和发送按钮布局
        entry_layout.addWidget(self.message_entry)
        entry_layout.addWidget(self.send_button)

        menu_layout = QVBoxLayout()  # 角色和情感选择布局
        menu_layout.addWidget(QLabel("选择角色：", self))
        menu_layout.addWidget(self.character_selector)
        menu_layout.addWidget(QLabel("选择情感：", self))
        menu_layout.addWidget(self.emotion_selector)

        main_layout = QVBoxLayout()  # 主布局
        main_layout.addLayout(menu_layout)
        main_layout.addWidget(self.chat_area)  # 添加聊天区域
        main_layout.addLayout(entry_layout)  # 添加输入框和按钮

        self.setLayout(main_layout)

        # 绑定消息接收信号到显示函数
        self.message_received.connect(self.add_message_bubble)

    def resizeEvent(self, event):
        """窗口大小变化时调整气泡宽度"""
        window_width = self.width()
        self.max_bubble_width = int(window_width * 1.2)
        super().resizeEvent(event)

    def add_message_bubble(self, message, position="left"):
        """在聊天区域添加消息气泡"""
        bubble_color = "#E5E5E5" if position == "left" else "#4CAF50"
        text_color = "#000000" if position == "left" else "#FFFFFF"

        # 创建气泡组件
        bubble = QLabel(message)
        bubble.setStyleSheet(
            f"background-color: {bubble_color}; color: {text_color}; font-size: 12pt; padding: 10px; "
            f"border-radius: 10px; max-width: {self.max_bubble_width}px;"
        )
        bubble.setWordWrap(True)  # 自动换行
        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 支持文本选择

        # 设置气泡布局
        bubble_layout = QHBoxLayout()
        if position == "left":
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()
        else:
            bubble_layout.addStretch()
            bubble_layout.addWidget(bubble)

        # 添加到聊天区域
        bubble_container = QFrame()
        bubble_container.setLayout(bubble_layout)

        self.chat_content.addWidget(bubble_container)
        self.chat_area.verticalScrollBar().setValue(self.chat_area.verticalScrollBar().maximum())

    def request_character_and_emotion(self):
        """请求服务器返回角色和情感列表"""
        self.client_socket.send("get_characters".encode('utf-8'))  # 请求角色和情感数据
        data = self.client_socket.recv(1024).decode('utf-8')  # 接收数据

        try:
            # 解析JSON格式的角色和情感数据
            characters_and_emotions = json.loads(data)  # 假设返回JSON格式
            self.character_selector.addItems(characters_and_emotions.keys())  # 添加角色选项

            # 默认选择第一个角色，更新对应情感
            if characters_and_emotions:
                self.character_selector.setCurrentIndex(0)
                self.update_emotions(characters_and_emotions)

            # 角色变更时动态更新情感选项
            self.character_selector.currentIndexChanged.connect(
                lambda: self.update_emotions(characters_and_emotions)
            )

        except json.JSONDecodeError:
            print("角色和情感数据解析失败")

    def update_emotions(self, characters_and_emotions):
        """根据选中角色更新情感下拉框"""
        selected_character = self.character_selector.currentText()
        emotions = characters_and_emotions.get(selected_character, [])  # 获取选中角色的情感
        self.emotion_selector.clear()
        self.emotion_selector.addItems(emotions)  # 更新情感选项

    def send_message(self):
        message = self.message_entry.text().strip()  # 获取输入框的消息
        if message:
            character = self.character_selector.currentText()  # 获取选中角色
            emotion = self.emotion_selector.currentText()  # 获取选中情感

            # 在聊天窗口显示消息
            self.add_message_bubble(f"{message}", "right")

            # 将消息发送给服务器
            formatted_message = f"{character},{emotion},{message}"
            self.client_socket.send(formatted_message.encode('utf-8'))

            self.message_entry.clear()  # 清空输入框

    def receive_message(self):
        """接收来自服务器的消息"""
        while True:
            try:
                message = self.client_socket.recv(1024).decode('utf-8')
                if message:
                    self.message_received.emit(message)  # 使用信号通知主线程更新界面
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        try:
            self.client_socket.send("close".encode('utf-8'))  # 通知服务器关闭连接
        except Exception as e:
            print(f"Error sending close message: {e}")
        self.client_socket.close()  # 关闭套接字
        event.accept()  # 接受关闭事件

if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建Qt应用
    client = ChatClient()  # 创建客户端窗口
    client.show()  # 显示窗口
    sys.exit(app.exec_())  # 运行应用程序
