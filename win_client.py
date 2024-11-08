import tkinter as tk
import socket
import threading

# 连接到服务器的函数
def connect_to_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 5555))  # 连接到本地的服务器
    return client_socket

# 发送消息的函数
def send_message():
    message = message_entry.get()
    if message.strip():  # 确保消息不是空白
        # 将消息显示在右侧（自己发送的消息）
        chat_window.insert(tk.END, create_message_bubble(message, "right"))
        chat_window.yview(tk.END)  # 滚动到最新的消息
        client_socket.send(message.encode('utf-8'))
        message_entry.delete(0, tk.END)

# 接收消息的函数
def receive_message():
    while True:
        try:
            message = client_socket.recv(1024).decode('utf-8')
            if message:
                # 将接收到的消息显示在左侧（对方发送的消息）
                chat_window.insert(tk.END, create_message_bubble(message, "left"))
                chat_window.yview(tk.END)  # 滚动到最新的消息
        except Exception as e:
            print(f"Error receiving message: {e}")
            break

# 创建消息气泡
def create_message_bubble(message, position):
    bubble_color = "#E5E5E5" if position == "left" else "#4CAF50"
    text_color = "#000000" if position == "left" else "#ffffff"
    
    # 设置消息气泡框架
    bubble = tk.Frame(chat_window, bg=bubble_color, relief="flat", bd=0, padx=10, pady=5)
    
    # 设置Label的wraplength来限制宽度并启用换行，设置anchor为"w"左对齐
    label = tk.Label(
        bubble,
        text=f"{message}",
        bg=bubble_color,
        fg=text_color,
        font=("Arial", 12),
        wraplength=250,  # 换行宽度
        justify="left",  # 文本左对齐
        anchor="w"  # 内容左对齐
    )
    label.pack(fill="both", expand=True)  # 使气泡填满Label并适应对齐方式
    
    # 显示消息气泡，左对齐或右对齐
    bubble.pack(anchor="w" if position == "left" else "e", padx=10, pady=5)
    return bubble


# GUI部分
root = tk.Tk()
root.title("微信样式聊天程序")

# 获取屏幕宽度和高度
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# 设置窗口大小
window_width = 400
window_height = 600

# 计算窗口位置，确保窗口居中
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)

# 设置窗口大小和位置
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
root.configure(bg="#F5F5F5")

# 聊天框，设置宽度和高度
chat_window = tk.Text(root, height=20, width=50, wrap=tk.WORD, bg="#F5F5F5", fg="#000000", font=("Arial", 12), bd=0, padx=10, pady=10)
chat_window.pack(pady=10, fill="both", expand=True)
chat_window.config(state=tk.DISABLED)  # 禁止编辑

# 消息输入框，设置字体、大小
message_entry = tk.Entry(root, width=40, font=("Arial", 12), bd=2, relief="sunken", fg="#000000")
message_entry.pack(pady=10)

# 发送按钮，设置颜色和字体
send_button = tk.Button(root, text="发送", command=send_message, font=("Arial", 12), bg="#4CAF50", fg="white", relief="flat", height=2, width=10)
send_button.pack(pady=10)

# 启动连接
client_socket = connect_to_server()

# 启动接收消息的线程
threading.Thread(target=receive_message, daemon=True).start()

# 客户端关闭时通知服务端
def on_closing():
    print("Closing the client...")
    client_socket.send("close".encode('utf-8'))  # 发送关闭消息到服务端
    client_socket.close()  # 关闭socket连接
    root.quit()  # 退出Tkinter主循环

# 将退出事件绑定到窗口关闭按钮
root.protocol("WM_DELETE_WINDOW", on_closing)

# 运行GUI
root.mainloop()
