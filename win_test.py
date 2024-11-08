import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QScrollArea, QTextEdit
from PyQt5.QtCore import Qt

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chat Application")
        self.setGeometry(100, 100, 400, 600)
        
        # 确保整个窗口的背景是透明的
        self.setStyleSheet("background-color: #f0f0f0; border-radius: 20px;")

        # 主布局
        main_layout = QVBoxLayout(self)

        # 聊天区域
        self.chat_area = QVBoxLayout()
        self.chat_widget = QWidget()
        self.chat_widget.setLayout(self.chat_area)

        # 滚动区域，用于显示聊天记录
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.chat_widget)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            background: transparent;
            border-radius: 20px;  # 让滚动区域有圆角
        """)
        main_layout.addWidget(self.scroll_area)

        # 输入框和发送按钮布局
        input_layout = QHBoxLayout()

        # 输入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setStyleSheet("""
            background-color: white;
            border-radius: 20px;
            padding: 10px 15px;
            font-size: 14px;
            border: 1px solid #ccc;
        """)  # 设置输入框圆角样式
        input_layout.addWidget(self.input_field)

        # 发送按钮
        send_button = QPushButton("Send")
        send_button.setFixedSize(80, 40)  # 强制设置按钮大小
        send_button.setStyleSheet("""
            background-color: #4CAF50;  # 按钮的背景颜色
            color: white;  # 按钮的文本颜色
            border-radius: 20px;  # 圆角效果
            font-size: 14px;  # 字体大小
            border: none;  # 去除默认的边框
        """)  # 设置按钮圆角样式
        send_button.clicked.connect(self.send_message)
        input_layout.addWidget(send_button)

        # 将输入框和按钮加入到主布局
        main_layout.addLayout(input_layout)

    def send_message(self):
        message = self.input_field.text()
        if message:
            self.add_chat_bubble(message, is_user=True)  # 用户消息
            self.input_field.clear()

            # 模拟一个对方的回复
            self.add_chat_bubble("This is Jon Snow's reply.", is_user=False)  # 模拟对方回复

    def add_chat_bubble(self, message, is_user=True):
        bubble_layout = QHBoxLayout()

        # 聊天气泡
        bubble = QTextEdit(message)
        bubble.setReadOnly(True)  # 使文本框内容不可编辑
        bubble.setStyleSheet("""
            background-color: #87CEFA;
            border-radius: 15px;
            color: white;
            padding: 10px;
            font-size: 14px;
            word-wrap: break-word;
            max-width: 250px;
            margin: 5px;
        """ if is_user else """
            background-color: #E6E6E6;
            border-radius: 15px;
            color: black;
            padding: 10px;
            font-size: 14px;
            word-wrap: break-word;
            max-width: 250px;
            margin: 5px;
        """)

        bubble.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 使文本可复制

        if is_user:
            bubble_layout.addStretch()
            bubble_layout.addWidget(bubble)
        else:
            bubble_layout.addWidget(bubble)
            bubble_layout.addStretch()

        chat_container = QWidget()
        chat_container.setLayout(bubble_layout)
        self.chat_area.addWidget(chat_container)

        # 自动滚动到底部
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
