'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, re, logging
import LangSegment
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch

# 检查并加载模型权重
if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", 'r', encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get(
            "gpt_path", gweight_data)
else:
    gpt_path = os.environ.get(
        "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", 'r', encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
# gpt_path = os.environ.get(
#     "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
# )
# sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")

# 设置环境变量
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
punctuation = set(['!', '?', '…', ',', '.', '-'," "])
# 导入gradio库和其他必要库
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()









# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。
# 检查是否有可用的CUDA设备，若有则使用GPU（"cuda"），否则使用CPU（"cpu"）
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 加载BERT模型和分词器
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

# 若is_half为True，将模型转换为半精度（float16），否则保持全精度（float32）
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

# 定义获取BERT特征的函数，输入为文本text和字音映射word2ph
def get_bert_feature(text, word2ph):
    # 禁用梯度计算，减少计算负担
    with torch.no_grad():
        # 将文本转为BERT输入格式（张量形式）
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        
        # 获取BERT模型的输出，提取隐藏状态
        res = bert_model(**inputs, output_hidden_states=True)
        # 拼接倒数第三层的隐藏状态（常用于特征提取）
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    
    # 确保输入的音素与文本长度相同
    assert len(word2ph) == len(text)
    
    # 逐字转换为对应音素的特征，并重复扩展为音素级别的特征
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    
    # 将特征拼接为最终的音素级特征矩阵，并返回其转置
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

# 定义递归字典类，将嵌套字典转换为具有属性访问方式的对象
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    # 实现属性访问方式
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    # 实现属性设置
    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    # 实现属性删除
    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# 加载CNHubert模型，并根据is_half切换精度
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

# 定义切换SoVITS模型权重的函数
def change_sovits_weights(sovits_path):
    global vq_model, hps
    # 加载SoVITS模型权重文件
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    
    # 创建并初始化SynthesizerTrn模型
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    
    # 若路径中不包含"pretrained"，则删除编码器部分
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    
    # 根据is_half设置模型精度
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    
    # 将模型设为评估模式并加载权重
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    
    # 将权重路径保存到文件中
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)

# 调用函数加载SoVITS模型权重
change_sovits_weights(sovits_path)

# 定义切换GPT模型权重的函数
def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    # 加载GPT模型权重文件
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    
    # 初始化Text2Semantic模型
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    
    # 根据is_half切换模型精度
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    
    # 打印模型参数数量
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    # 将GPT权重路径保存到文件中
    with open("./gweight.txt", "w", encoding="utf-8") as f: 
        f.write(gpt_path)

# 调用函数加载GPT模型权重
change_gpt_weights(gpt_path)

# 定义获取频谱图的函数
def get_spepc(hps, filename):
    # 加载音频文件并转换为张量
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio.unsqueeze(0)
    
    # 计算并返回音频的频谱图
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec

# 定义语言字典，将不同语言选项映射到模型标签
dict_language = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别
    i18n("日英混合"): "ja",  # 按日英混合识别
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}

# 清理并格式化输入文本函数，返回处理后的音素、音素映射及规范化文本
def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

# 设置模型的默认数据类型为float16或float32
dtype = torch.float16 if is_half == True else torch.float32

# 定义BERT特征提取的函数，若语言为中文则调用BERT，否则返回全零矩阵
def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)
    return bert

# 定义分割符号集合，用于文本分句
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

# 定义获取文本首句的函数
def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text










# 定义获取音素和BERT特征的函数
def get_phones_and_bert(text, language):
    # 如果语言为英文或全中文、全日文，首先统一为简化语言代码
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")  # 去掉"all_"前缀
        
        if language == "en":  
            # 若语言为英文，设置过滤器只处理英文
            LangSegment.setfilters(["en"])
            # 通过语言分段工具获取文本并组合成格式化文本
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 如果是中文或日文，直接使用输入的文本作为格式化文本
            formattext = text
        
        # 去除多余的空格
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        
        # 清理文本并获取音素、音素到词的映射及规范化后的文本
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        
        # 如果语言为中文，调用BERT特征提取函数
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            # 否则生成一个全零的张量作为BERT特征（比如英文情况下不使用BERT）
            bert = torch.zeros(
                (1024, len(phones)),  # 1024维向量，长度为音素的数量
                dtype=torch.float16 if is_half == True else torch.float32
            ).to(device)
    
    # 处理中文、日文和自动语言识别的情况
    elif language in {"zh", "ja", "auto"}:
        textlist = []  # 存储分段后的文本
        langlist = []  # 存储对应的语言

        # 设置多语言过滤器（包括中文、日文、英文、韩文）
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        
        # 如果选择自动识别语言
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":  
                    # 将韩文识别为中文处理
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    # 记录检测到的语言和对应的文本
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            # 如果没有自动识别，则根据用户指定的语言来处理
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":  
                    # 如果是英文，保留语言为英文
                    langlist.append(tmp["lang"])
                else:
                    # 如果是中文或日文，按照用户输入的语言处理
                    langlist.append(language)
                textlist.append(tmp["text"])
        
        # 打印分段后的文本和对应的语言
        print(textlist)
        print(langlist)

        phones_list = []  # 存储所有段落的音素
        bert_list = []    # 存储所有段落的BERT特征
        norm_text_list = []  # 存储所有段落的规范化文本

        # 遍历每一段文本并根据语言分别处理
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            # 获取每一段的BERT特征
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        
        # 将所有BERT特征拼接在一起，沿列方向进行拼接
        bert = torch.cat(bert_list, dim=1)
        # 将所有音素列表合并
        phones = sum(phones_list, [])
        # 将所有规范化文本拼接为一个完整文本
        norm_text = ''.join(norm_text_list)

    # 返回音素、BERT特征和规范化文本
    return phones, bert.to(dtype), norm_text

# 定义合并短文本的函数
def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result










# 定义语音合成的函数
def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, 
                how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free=False):
    
    try:
        top_k = int(top_k)
        top_p = float(top_p)
        temperature = float(temperature)
    except ValueError as e:
        raise ValueError(f"Invalid value for parameters: top_k={top_k}, top_p={top_p}, temperature={temperature}.") from e

    # 如果没有提供 prompt_text 或其长度为 0，启用自由合成模式 (ref_free)
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    # 记录当前时间戳 t0，用于性能测量
    t0 = ttime()

    # 将语言标签转换为内部字典使用的语言编码
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    # 如果不是自由合成，处理 prompt_text
    if not ref_free:
        prompt_text = prompt_text.strip("\n")  # 去除换行符
        if (prompt_text[-1] not in splits):  # 检查句末是否有标点符号
            prompt_text += "。" if prompt_language != "en" else "."  # 根据语言添加句号或句点
        print(i18n("实际输入的参考文本:"), prompt_text)

    # 处理目标文本，去除换行符并规范化标点
    text = text.strip("\n")
    text = replace_consecutive_punctuation(text)
    # 如果目标文本没有以标点开头且长度小于 4，则自动补上标点符号
    if (text[0] not in splits and len(get_first(text)) < 4): 
        text = "。" + text if text_language != "en" else "." + text
    
    print(i18n("实际输入的目标文本:"), text)

    # 创建一个零音频段（长度为0.3秒），用于在合成音频之间插入短暂停顿
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )

    # 如果不是自由合成模式，需要加载参考音频并提取语义编码,ref_free 控制是否使用参考音频
    if not ref_free:
        with torch.no_grad():
            # 加载参考音频文件，采样率为 16000 Hz
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            
            # 如果参考音频长度不在 3 到 10 秒之间，抛出错误
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            
            # 将 numpy 数组转换为 torch 张量
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            
            # 如果使用半精度 (float16)，则将音频张量转换为半精度
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)

            # 在参考音频后拼接零音频段
            wav16k = torch.cat([wav16k, zero_wav_torch])
            
            # 使用 SSL 模型提取参考音频的最后隐状态作为特征
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            # 提取 VQ-VAE 编码
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]  # 取出第一帧的编码
            prompt = prompt_semantic.unsqueeze(0).to(device)  # 转换为设备张量

    # 记录处理参考音频所花的时间
    t1 = ttime()

    # 根据用户选择的切割方式，切割文本
    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    
    # 删除多余的换行符
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    
    print(i18n("实际输入的目标文本(切句后):"), text)

    # 按换行符拆分文本段落，并对短文本段落进行合并
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)  # 将小于5个字符的文本合并

    # 存储生成的音频数据
    audio_opt = []

    # 如果不是自由合成，获取参考文本的音素和BERT特征
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language)

    # 遍历每一个目标文本段落
    for text in texts:
        # 忽略空行
        if (len(text.strip()) == 0):
            continue
        # 补充句末的标点符号
        if (text[-1] not in splits): 
            text += "。" if text_language != "en" else "."
        
        print(i18n("实际输入的目标文本(每句):"), text)
        
        # 获取目标文本的音素和BERT特征
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        
        # 如果参考文本可用，将参考文本和目标文本的 BERT 特征合并
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        # 准备 BERT 特征和音素长度的张量
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        # 记录生成音频前的时间戳 t2
        t2 = ttime()
        
        with torch.no_grad():
            # 使用 TTS 模型推理语义编码
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,  # 如果是自由合成，不使用参考提示
                bert,
                top_k=top_k,  # 影响生成的多样性
                top_p=top_p,  # 影响生成的多样性
                temperature=temperature,  # 控制生成概率的温度
                early_stop_num=hz * max_sec,  # 生成的早停机制
            )
        
        # 记录推理所花的时间
        t3 = ttime()

        # 剪切生成的语义编码
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        
        # 获取参考音频的声谱特征
        refer = get_spepc(hps, ref_wav_path)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)

        # 使用 VQ-VAE 解码生成语音
        audio = vq_model.decode(
            pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
        ).detach().cpu().numpy()[0, 0]
        
        # 预防音频过载，归一化音频
        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
        
        # 将生成的音频加入输出音频列表，并在片段之间插入零音频
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        
        # 记录解码所花的时间
        t4 = ttime()
    
    # 打印各个阶段所花的时间
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))

    # 返回采样率及最终合成音频
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)













def split(todo_text):
    # 将 "……" 替换为句号，将 "——" 替换为逗号，便于后续统一处理
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    
    # 如果文本最后一个字符不是标点符号，自动在末尾加上句号
    if todo_text[-1] not in splits:
        todo_text += "。"
    
    # 初始化两个指针，分别用于标记分割的开始位置和结束位置
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)  # 文本的长度
    todo_texts = []  # 存放分割后的文本片段的列表
    
    # 循环遍历文本，直到遍历到文本末尾
    while 1:
        if i_split_head >= len_text:
            break  # 如果遍历到文本末尾，结束循环
        
        # 如果当前位置是标点符号
        if todo_text[i_split_head] in splits:
            i_split_head += 1  # 将分割位置移动到下一个字符
            # 将从上一个分割点到当前分割点的文本片段添加到列表中
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head  # 更新分割开始位置为当前的位置
        else:
            i_split_head += 1  # 如果不是标点符号，继续遍历
    
    # 返回分割后的文本片段列表
    return todo_texts

def cut1(inp):
    inp = inp.strip("\n")  # 去除输入文本中的换行符
    inps = split(inp)  # 调用 split 函数进行分割
    split_idx = list(range(0, len(inps), 4))  # 每四个句子分割一次
    split_idx[-1] = None  # 设置最后一段的结束位置为 None（即文本末尾）
    
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            # 将四个句子合并成一个片段
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]  # 如果文本很短，直接返回原文
    
    # 删除仅由标点符号组成的片段
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    
    return "\n".join(opts)  # 以换行符连接每个片段


def cut2(inp):
    inp = inp.strip("\n")  # 去除输入文本中的换行符
    inps = split(inp)  # 调用 split 函数进行分割
    if len(inps) < 2:
        return inp  # 如果分割后的段落少于两个，直接返回原文
    
    opts = []
    summ = 0  # 统计当前段落的总字符数
    tmp_str = ""  # 用于临时存储当前段落的内容
    
    for i in range(len(inps)):
        summ += len(inps[i])  # 累计段落的字符数
        tmp_str += inps[i]  # 累计段落的内容
        
        if summ > 50:  # 当段落字符数超过50时，保存该段落并重置累加器
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    
    if tmp_str != "":  # 如果最后还有未处理的段落，添加到结果中
        opts.append(tmp_str)
    
    # 如果最后一个段落较短，将它和前一个段落合并
    if len(opts) > 1 and len(opts[-1]) < 50:
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    
    # 删除仅由标点符号组成的片段
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    
    return "\n".join(opts)  # 以换行符连接每个片段


def cut3(inp):
    inp = inp.strip("\n")  # 去除文本中的换行符
    # 按中文句号 '。' 分割文本，并去掉首尾的句号（若有）
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    # 删除只包含标点符号的片段
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    # 将分割后的有效片段用换行符连接并返回
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")  # 去除文本中的换行符
    # 按英文句号 '.' 分割文本，并去掉首尾的句号（若有）
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    # 删除只包含标点符号的片段
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    # 将分割后的有效片段用换行符连接并返回
    return "\n".join(opts)


def cut5(inp):
    inp = inp.strip("\n")  # 去除文本中的换行符
    # 定义各种标点符号的集合
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []  # 用于存储最终分割后的文本片段
    items = []  # 用于暂时存储每一片段

    # 遍历文本中的每个字符
    for i, char in enumerate(inp):
        if char in punds:  # 如果字符是标点符号
            # 特殊处理：如果当前字符是小数点且前后都是数字，则视为数字的一部分
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)  # 将标点符号加入当前片段
                mergeitems.append("".join(items))  # 将当前片段合并并存储
                items = []  # 清空片段
        else:
            items.append(char)  # 如果是普通字符，加入当前片段

    if items:  # 如果还有未处理的片段，存储
        mergeitems.append("".join(items))

    # 删除只包含标点符号的片段
    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    
    # 将结果用换行符连接返回
    return "\n".join(opt)


##### 对包含数字和非数字部分的字符串进行排序。它将字符串中的数字提取出来，并将其转换为整数，以实现更加自然的排序（即按数值大小排序，而不是按字符顺序
def custom_sort_key(s):
    # 使用正则表达式将字符串中的数字和非数字部分分离
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持原样
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts  # 返回分离后的部分，用于排序


##### 清理输入文本列表，删除空白、无效的文本片段。
def process_text(texts):
    _text = []  # 存储有效文本的列表
    # 如果所有输入的文本都是无效（空白、None等），则抛出异常
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    
    # 遍历文本，过滤掉无效文本
    for text in texts:
        if text not in [None, " ", ""]:
            _text.append(text)
    
    return _text  # 返回有效文本的列表


##### 替换连续出现的标点符号，只保留第一个标点。
def replace_consecutive_punctuation(text):
    # 获取所有标点符号并进行转义，确保正则表达式处理
    punctuations = ''.join(re.escape(p) for p in punctuation)
    # 正则表达式模式，用于匹配连续出现的标点符号
    pattern = f'([{punctuations}])([{punctuations}])+'
    # 用正则表达式替换连续出现的标点符号
    result = re.sub(pattern, r'\1', text)
    return result  # 返回处理后的文本


##### 获取预训练模型的名称，并按自然顺序排序。
def change_choices():
    # 获取 SoVITS 和 GPT 模型的名称
    SoVITS_names, GPT_names = get_weights_names()
    # 返回排序后的模型名称列表
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, \
           {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


##### 定义 SoVITS 和 GPT 模型的预训练权重文件路径。
##### 使用 os.makedirs 创建存放 SoVITS 和 GPT 权重文件的目录，若不存在则创建。
pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)











##### 该函数用于获取两个文件夹（SoVITS_weight_root 和 GPT_weight_root）中的模型权重文件名，分别返回 SoVITS 和 GPT 模型的名称列表。
def get_weights_names():
    # 初始化 SoVITS 模型名称列表，首先包含预训练模型的路径
    SoVITS_names = [pretrained_sovits_name]
    # 遍历 SoVITS 模型存放的目录
    for name in os.listdir(SoVITS_weight_root):
        # 如果文件名以 ".pth" 结尾（代表是模型权重文件）
        if name.endswith(".pth"):
            # 将模型文件路径加入 SoVITS_names 列表
            SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    
    # 初始化 GPT 模型名称列表，首先包含预训练模型的路径
    GPT_names = [pretrained_gpt_name]
    # 遍历 GPT 模型存放的目录
    for name in os.listdir(GPT_weight_root):
        # 如果文件名以 ".ckpt" 结尾（代表是 GPT 模型权重文件）
        if name.endswith(".ckpt"):
            # 将模型文件路径加入 GPT_names 列表
            GPT_names.append("%s/%s" % (GPT_weight_root, name))
    
    # 返回两个列表：SoVITS 模型列表和 GPT 模型列表
    return SoVITS_names, GPT_names

SoVITS_names, GPT_names = get_weights_names()


##### gr.Blocks 是 Gradio 的核心组件，用于构建交互式的用户界面。
with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    # 声明软件使用 MIT 协议，并提醒用户需对软件使用负全责
    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )

    with gr.Group():
        # 模型切换部分
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            # GPT 模型下拉列表
            GPT_dropdown = gr.Dropdown(
                label=i18n("GPT模型列表"),
                choices=sorted(GPT_names, key=custom_sort_key),  # 选择 GPT 模型列表
                value=gpt_path,  # 默认值为 gpt_path
                interactive=True
            )
            # SoVITS 模型下拉列表
            SoVITS_dropdown = gr.Dropdown(
                label=i18n("SoVITS模型列表"),
                choices=sorted(SoVITS_names, key=custom_sort_key),  # 选择 SoVITS 模型列表
                value=sovits_path,  # 默认值为 sovits_path
                interactive=True
            )
            # 刷新按钮，点击刷新模型路径
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

            # 切换 SoVITS 或 GPT 模型时触发相应的操作
            SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
            GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

        # 上传参考音频部分
        gr.Markdown(value=i18n("*请上传并填写参考信息"))
        with gr.Row():
            # 上传参考音频，限制时长在 3-10 秒以内
            inp_ref = gr.Audio(
                label=i18n("请上传3~10秒内参考音频，超过会报错！"),
                type="filepath"
            )
            with gr.Column():
                # 选择无参考文本模式
                ref_text_free = gr.Checkbox(
                    label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"),
                    value=False,
                    interactive=True,
                    show_label=True
                )
                # 参考音频的文本输入框
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="")
            
            # 参考音频的语言选择
            prompt_language = gr.Dropdown(
                label=i18n("参考音频的语种"),
                choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")],
                value=i18n("中文")
            )
        
            # 填写需要合成的目标文本部分
            gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
            with gr.Row():
                # 需要合成的目标文本输入框
                text = gr.Textbox(label=i18n("需要合成的文本"), value="")
                
                # 选择目标文本的语种
                text_language = gr.Dropdown(
                    label=i18n("需要合成的语种"),
                    choices=[i18n("中文")],  # 仅保留中英混合选项
                    value=i18n("中文"),  # 设置默认值为中英混合
                    interactive=False  # 不允许用户更改
                )
                
                # 文本切分方式选择
                how_to_cut = gr.Radio(
                    label=i18n("怎么切"),
                    choices=[i18n("不切")],  # 仅保留按标点符号切选项
                    value=i18n("不切"),  # 设置默认值为按标点符号切
                    interactive=False  # 不允许用户更改
                )
            
            # gpt采样参数设置
            with gr.Row():
                gr.Markdown(value=i18n("gpt采样参数(无参考文本时不要太低)："))
                
                # 替换 Slider 为 Textbox
                top_k = gr.Textbox(
                    label=i18n("top_k"),
                    value="5",  # 默认值为 5，范围 0-100
                    interactive=True    # interactive 属性设置为 True，允许用户在运行时进行交互。
                )
                
                top_p = gr.Textbox(
                    label=i18n("top_p"),
                    value="1",  # 默认值为 1，范围 0-1
                    interactive=True
                )
                
                temperature = gr.Textbox(
                    label=i18n("temperature"),
                    value="1",  # 默认值为 1，范围 0-1
                    interactive=True
                )
                
            # 合成语音按钮
            inference_button = gr.Button(i18n("合成语音"), variant="primary")
            output = gr.Audio(label=i18n("输出的语音"))

        # 点击合成按钮时调用 `get_tts_wav` 函数，传入所需参数并输出语音结果
        inference_button.click(
            get_tts_wav,
            [inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut, top_k, top_p, temperature, ref_text_free],
            [output]
        )
        

# 主函数，启动 Gradio 应用
if __name__ == '__main__':
    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
        quiet=True,
    )
