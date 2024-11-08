import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Chat Example")
        self.setGeometry(100, 100, 400, 600)

        # 设置背景透明和样式
        self.setStyleSheet("""
            QWidget {
                background: transparent;
            }
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 15px;
                padding: 10px;
                font-size: 14px;
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 15px;
                font-size: 14px;
                padding: 10px;
            }
            QLabel {
                padding: 10px;
                border-radius: 15px;
                background-color: #e1f5fe;
                font-size: 14px;
                max-width: 300px;
                word-wrap: break-word;
                margin: 5px;
            }
            QLabel#reply {
                background-color: #cfd8dc;
                color: black;
            }
            QScrollArea {
                border: none;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout(self)

        # 聊天区域
        self.chat_layout = QVBoxLayout()
        self.chat_widget = QWidget()
        self.chat_widget.setLayout(self.chat_layout)

        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.chat_widget)
        main_layout.addWidget(scroll_area)

        # 输入框和发送按钮
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setFixedHeight(50)

        send_button = QPushButton("Send", self)
        send_button.setFixedHeight(50)
        send_button.clicked.connect(self.sendMessage)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(send_button)

        main_layout.addLayout(input_layout)

    def sendMessage(self):
        message = self.input_field.text()
        if message:
            self.addChatBubble(message, is_user=True)  # 添加用户消息
            self.input_field.clear()  # 清空输入框

            # 模拟回复
            self.addChatBubble("This is a reply.", is_user=False)  # 添加回复消息

    def addChatBubble(self, message, is_user=True):
        bubble = QLabel(message)
        if not is_user:
            bubble.setObjectName("reply")  # 设置样式为对方消息

        self.chat_layout.addWidget(bubble)

        # 自动滚动到底部
        self.chat_widget.setLayout(self.chat_layout)  # 更新布局
        QTimer.singleShot(0, lambda: self.scroll_to_bottom())

    def scroll_to_bottom(self):
        scroll_area = self.findChild(QScrollArea)
        scroll_bar = scroll_area.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())  # 滚动到最底部


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())
