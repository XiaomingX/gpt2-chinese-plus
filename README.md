---

# GPT2-中文版

## 描述

- 这是GPT2的中文训练代码，支持BERT分词器或BPE分词器。基于HuggingFace团队的[Transformers](https://github.com/huggingface/transformers)仓库，可以生成诗歌、新闻、小说，或训练通用语言模型。支持字符级、词级和BPE级别的训练，支持大规模语料。
- 中文版GPT2训练代码，使用BERT的分词器或Sentencepiece的BPE模型（感谢[kangzhonghua](https://github.com/kangzhonghua)的贡献，使用BPE模式时需要稍微修改`train.py`文件）。可以生成诗歌、新闻、小说等文本，也可以用于通用语言模型的训练。支持字、词和BPE三种分词模式（需要稍微修改`train.py`文件），支持大规模语料。

## 更新说明

### 2024年11月4日

- 感谢大家对本项目的关注。自ChatGPT发布以来，该项目重新引起了一些关注。项目初衷是作为自学Pytorch的练手项目，因此没有长期维护更新的计划。
- 如有兴趣讨论大模型（LLM），可以通过邮箱(ned1991@gmail.com)与我联系，或在Issue中交流。

### 2021年6月2日

- 本项目新增了[通用中文GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#模型分享)、[通用中文GPT-2小模型](https://github.com/Morizeyao/GPT2-Chinese#模型分享)、[中文歌词GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#模型分享)和[文言文GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#模型分享)。模型由UER-py项目训练，已上传到HuggingFace Model Hub，更多细节可参考以下链接：[gpt2-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)、[gpt2-distil-chinese-cluecorpussmall](https://huggingface.co/uer/gpt2-distil-chinese-cluecorpussmall)、[gpt2-chinese-lyric](https://huggingface.co/uer/gpt2-chinese-lyric)和[gpt2-chinese-ancient](https://huggingface.co/uer/gpt2-chinese-ancient)。  
- 生成时需在输入文本前加入起始符，例如输入“最美的不是下雨天，是曾与你躲过雨的屋檐”时，格式为“[CLS]最美的不是下雨天，是曾与你躲过雨的屋檐”。

## 更新内容

### 2020年11月3日

- 本项目新增了[古诗词GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#模型分享)和[对联GPT-2预训练模型](https://github.com/Morizeyao/GPT2-Chinese#模型分享)，模型由UER-py项目训练并上传至HuggingFace Model Hub。  
- 使用时需在输入文本前加“[CLS]”，如输入“梅山如积翠，”，格式为“[CLS]梅山如积翠，”；对联模型训练的格式为“上联-下联”，生成时输入“丹枫江冷人初去-”，格式为“[CLS]丹枫江冷人初去-”。

---

## 使用方法

1. 在项目根目录下建立`data`文件夹，并将训练语料命名为`train.json`放入该文件夹。**`train.json`文件中每个元素为一篇训练文本的内容（而非文件路径）**。
2. 运行`train.py`文件，勾选`--raw`参数将自动预处理数据。
3. 预处理完成后自动开始训练。

---

## 文件说明

- `generate.py` 和 `train.py` 分别是生成与训练脚本。
- `train_single.py` 可用于处理大的单一文本数据集（例如训练一本小说）。
- `eval.py` 用于计算生成模型的困惑度（ppl）。
- `generate_texts.py` 是`generate.py`的扩展，可生成包含特定起始关键词的多句文本并保存到文件。
- `train.json` 是训练样本的格式示例文件。
- `cache` 文件夹包含若干BERT词表，`make_vocab.py`用于在语料上构建自定义词表。
- `tokenizations` 文件夹包含三种可选分词器：BERT分词器、分词版BERT分词器和BPE分词器。
- `scripts` 文件夹包含示例训练和生成脚本。
