import torch
print(torch.cuda.is_available())  # 检查CUDA是否可用,False表示PyTorch不支持CUDA 12.1
print(torch.backends.cudnn.enabled)  # 检查cuDNN是否可用