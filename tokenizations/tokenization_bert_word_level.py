# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import os
import unicodedata
import thulac
from io import open
from transformers.tokenization_utils import PreTrainedTokenizer

# 初始化thulac用于中文分词，只进行分词，不进行词性标注
lac = thulac.thulac(user_dict='tokenizations/thulac_dict/seg', seg_only=True)

# 预训练词汇文件的名称
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

# 预训练模型与对应词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file': {
        'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        'bert-base-german-cased': "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt"
    }
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
    'bert-base-german-cased': 512,
    'bert-large-uncased-whole-word-masking': 512,
    'bert-large-cased-whole-word-masking': 512,
    'bert-large-uncased-whole-word-masking-finetuned-squad': 512,
    'bert-large-cased-whole-word-masking-finetuned-squad': 512,
    'bert-base-cased-finetuned-mrpc': 512
}

def load_vocab(vocab_file):
    """
    从词汇文件中加载词汇表。
    Args:
        vocab_file: 词汇文件路径。
    Returns:
        vocab: 词汇表，键为词汇，值为索引。
    """
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        for index, token in enumerate(reader.readlines()):
            vocab[token.strip()] = index
    return vocab

def whitespace_tokenize(text):
    """
    对文本进行简单的空格分割。
    Args:
        text: 输入文本。
    Returns:
        tokens: 分割后的词汇列表。
    """
    text = text.strip()
    return text.split() if text else []

class BertTokenizer(PreTrainedTokenizer):
    """
    BERT分词器，提供基本分词和WordPiece分词功能。
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        super().__init__(unk_token=unk_token, sep_token=sep_token,
                         pad_token=pad_token, cls_token=cls_token,
                         mask_token=mask_token, **kwargs)
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = {idx: tok for tok, idx in self.vocab.items()}
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split,
                                                  tokenize_chinese_chars=tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def _tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                split_tokens.extend(self.wordpiece_tokenizer.tokenize(token))
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens).replace(' ##', '').strip()

class BasicTokenizer:
    """
    基本分词器，支持标点符号分割、小写转换等功能。
    """
    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text):
        text = self._clean_text(text)
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in tokens:
            if self.do_lower_case and token not in self.never_split:
                token = self._run_strip_accents(token.lower())
            split_tokens.extend(self._run_split_on_punc(token))
        return whitespace_tokenize(" ".join(split_tokens))

    def _run_strip_accents(self, text):
        """ 去掉文本中的重音符号 """
        return ''.join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")

    def _run_split_on_punc(self, text):
        """ 将标点符号与其他字符分开 """
        if text in self.never_split:
            return [text]
        return [char if _is_punctuation(char) else char for char in text]

    def _tokenize_chinese_chars(self, text):
        """ 对中文字符添加空格，便于后续分词 """
        output = []
        for char in text:
            if char.isdigit() or self._is_chinese_char(ord(char)):
                output.append(f" {char} ")
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """ 检查字符是否为CJK字符 """
        return (0x4E00 <= cp <= 0x9FFF or
                0x3400 <= cp <= 0x4DBF or
                0x20000 <= cp <= 0x2A6DF or
                0x2A700 <= cp <= 0x2B73F or
                0x2B740 <= cp <= 0x2B81F or
                0x2B820 <= cp <= 0x2CEAF or
                0xF900 <= cp <= 0xFAFF or
                0x2F800 <= cp <= 0x2FA1F)

    def _clean_text(self, text):
        """ 清理文本中的控制字符和无效字符 """
        return ''.join(char if not _is_control(char) else ' ' for char in text)

class WordpieceTokenizer:
    """
    WordPiece分词器，采用贪婪最长匹配算法。
    """
    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue
            sub_tokens = []
            start = 0
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = ''.join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    sub_tokens = [self.unk_token]
                    break
                sub_tokens.append(cur_substr)
                start = end
            output_tokens.extend(sub_tokens)
        return output_tokens

def _is_whitespace(char):
    """ 判断字符是否为空白字符 """
    return char in " \t\n\r" or unicodedata.category(char) == "Zs"

def _is_control(char):
    """ 判断字符是否为控制字符 """
    return unicodedata.category(char).startswith("C") and char not in "\t\n\r"

def _is_punctuation(char):
    """ 判断字符是否为标点符号 """
    cp = ord(char)
    return (33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126 or unicodedata.category(char).startswith("P"))
