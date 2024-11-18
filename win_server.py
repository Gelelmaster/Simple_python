import socket
import threading
import asyncio
import logging
from Gptsovit_tts import get_characters_and_emotions, text_to_speech
from Run_model import get_response  # 导入 get_response 函数

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 存储当前连接的客户端
clients = []
# 用于判断是否所有客户端都已断开连接
shutdown_flag = False
# 服务端是否正在关闭的标志
server_shutting_down = False

# 处理客户端消息的函数
async def handle_client(client_socket, addr):
    """处理客户端的消息并生成AI响应"""
    try:
        logger.info(f"New connection from {addr}")
        while True:
            message = client_socket.recv(1024).decode('utf-8')
            if not message:  # 如果客户端断开连接
                break
            
            logger.info(f"Received message: {message}")

            # 解析角色、情感和消息
            try:
                character, emotion, user_message = message.split(',', 2)
            except ValueError:
                # 如果无法分割为三个部分，认为是无效的消息格式
                logger.error("Invalid message format received from client.")
                continue
            
            logger.info(f"角色: {character}, 情感: {emotion}, 消息: {user_message}")

            # 获取AI的响应
            logger.info("\n正在调用模型生成回复...")
            response = await get_response(user_message)  # 使用异步获取AI的回复
            logger.info(f"AI回复: {response}")

            # 使用TTS模块进行语音播放
            logger.info("\n生成的模型回复将以语音输出...")
            await text_to_speech(response, character, emotion)

            # 将AI回复发送给客户端
            client_socket.send(response.encode('utf-8'))

    except Exception as e:
        logger.error(f"Error handling client {addr}: {e}")
    finally:
        # 客户端断开连接，清理工作
        logger.info(f"Closing connection with {addr}")
        clients.remove(client_socket)
        client_socket.close()

        global shutdown_flag
        # 如果没有剩余的客户端连接，则设置关闭标志
        if len(clients) == 0:
            shutdown_flag = True


# 创建并启动服务器
def start_server():
    global server_shutting_down
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 5555))  # 本地地址和端口
    server.listen(5)  # 最多连接5个客户端
    print("Server is listening on 127.0.0.1:5555...")

    try:
        while True:
            if server_shutting_down and shutdown_flag:  # 所有客户端都断开连接，退出服务器
                print("All clients are disconnected. Shutting down server...")
                break

            client_socket, addr = server.accept()
            clients.append(client_socket)  # 添加到连接的客户端列表
            threading.Thread(target=asyncio.run, args=(handle_client(client_socket, addr),)).start()
    except KeyboardInterrupt:
        print("Server is shutting down...")
    finally:
        # 关闭所有客户端连接
        for client in clients:
            client.close()
        server.close()

if __name__ == "__main__":
    start_server()
