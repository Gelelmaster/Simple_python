import pygame
import sys

# 初始化 Pygame
pygame.init()

# 初始化混音器
pygame.mixer.init()

# 加载音频文件（请替换为你的音频文件路径）
audio_file = r"D:\Desktop\project\GPT-Sovits-MHY\temp_audio_1729717041.wav"  # 确保这个文件存在
pygame.mixer.music.load(audio_file)

# 播放音频
pygame.mixer.music.play(-1)  # -1 表示循环播放

# 创建一个窗口
screen = pygame.display.set_mode((400, 150))  # 增加窗口大小
pygame.display.set_caption("音频播放器")

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # 空格键暂停/继续
                if pygame.mixer.music.get_busy():  # 如果音频正在播放
                    pygame.mixer.music.pause()
                else:
                    pygame.mixer.music.unpause()
            elif event.key == pygame.K_s:  # 's' 键停止
                pygame.mixer.music.stop()
            elif event.key == pygame.K_ESCAPE:  # ESC 键退出
                running = False

    # 更新屏幕
    screen.fill((0, 0, 0))  # 填充黑色背景
    font = pygame.font.Font(None, 36)
    text = font.render("按空格键暂停/继续, 'S'键停止, ESC键退出", True, (255, 255, 255))
    screen.blit(text, (10, 50))
    pygame.display.flip()

# 退出 Pygame
pygame.quit()
sys.exit()
