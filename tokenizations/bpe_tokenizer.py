# 导入必要的库
import json
import os
import sentencepiece as spm

# 获取一个单词中所有相邻字符对的集合
def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# BPE 编码器类
class Encoder:
    def __init__(self, encoder, bpe_merges):
        # 初始化编码器、解码器和BPE合并规则
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

    def bpe(self, token):
        # 如果缓存中存在该token，直接返回
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token

        # 不断合并频率最高的bigram，直到无法合并
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        
        # 将合并后的单词加入缓存
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        # 将文本编码为对应的ID列表
        return [self.encoder.get(token, 1) for token in self.tokenize(text)]

    def decode(self, tokens):
        # 将ID列表解码为文本
        return ''.join([self.decoder[token] for token in tokens])

    def tokenize(self, text):
        # 将文本进行BPE分词
        return self.bpe(text).split(' ')

    def convert_tokens_to_ids(self, tokens):
        # 将token列表转换为ID列表
        return [self.encoder.get(token, 1) for token in tokens]

# SentencePiece 编码器类
class EncoderSP:
    def __init__(self, model_path):
        # 加载SentencePiece模型
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        # 将文本编码为ID列表
        return self.sp.EncodeAsIds(text)

    def decode(self, tokens):
        # 将ID列表解码为文本
        return self.sp.DecodeIds([int(token) for token in tokens])

    def tokenize(self, text):
        # 将文本分割为token
        return self.sp.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        # 将token列表转换为ID列表
        return [self.sp.PieceToId(token) for token in tokens]

# 获取编码器函数
def get_encoder(encoder_file, bpe_file):
    # 根据文件扩展名判断使用哪种编码器
    _, extension = os.path.splitext(encoder_file)
    
    if extension == ".model" and not bpe_file:
        return EncoderSP(encoder_file)
    else:
        with open(encoder_file, 'r', encoding="utf-8") as f:
            encoder = json.load(f)
        with open(bpe_file, 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(encoder=encoder, bpe_merges=bpe_merges)
