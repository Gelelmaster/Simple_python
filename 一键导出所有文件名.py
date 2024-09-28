import os
import pandas as pd

def export_files(directory, file_format, extensions):
    try:
        files = os.listdir(directory)
    except Exception as e:
        print(f"无法读取目录: {e}")
        return

    # 将所有的扩展名转为小写，不区分大小写
    extensions = [ext.lower() for ext in extensions]

    # 筛选符合指定扩展名的文件
    filtered_files = [
        file for file in files 
        if file.lower().endswith(tuple(extensions)) and any(
            '\u4e00' <= char <= '\u9fff' or # 中文字符
            '\u3040' <= char <= '\u30ff' or # 日文平假名/片假名
            '\u0000' <= char <= '\u007f'    # 英文字母和数字
            for char in file)
    ]

    if file_format == 'csv':
        df = pd.DataFrame(filtered_files, columns=['文件名'])
        df.to_csv('exported_files.csv', index=False, encoding='utf-8-sig')
    elif file_format == 'txt':
        with open('exported_files.txt', 'w', encoding='utf-8') as f:
            for file in filtered_files:
                f.write(file + '\n')
    else:
        print("不支持的格式。请选择 'csv' 或 'txt'。")

# 使用示例
directory = input("请输入目录路径：")
file_format = input("选择导出格式 (csv/txt)：")
extensions_input = input("请输入要筛选的文件扩展名（用逗号分隔，例如 txt,jpg,png）：")
extensions = [ext.strip() for ext in extensions_input.split(",")]

export_files(directory, file_format, extensions)
