import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel

# 设置环境变量，指定程序使用的显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 判断是否为英文单词
# 这里的逻辑是判断单词中所有字母是否都在英文字母表内
def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

# 判断字符是否为中文字符
# 通过Unicode编码判断字符是否位于CJK（中日韩统一表意文字）范围内
def _is_chinese_char(char):
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  # CJK统一表意符号
            (cp >= 0x3400 and cp <= 0x4DBF) or  # CJK扩展A
            (cp >= 0x20000 and cp <= 0x2A6DF) or  # CJK扩展B
            (cp >= 0x2A700 and cp <= 0x2B73F) or  # CJK扩展C
            (cp >= 0x2B740 and cp <= 0x2B81F) or  # CJK扩展D
            (cp >= 0x2B820 and cp <= 0x2CEAF) or  # CJK扩展E
            (cp >= 0xF900 and cp <= 0xFAFF) or  # 兼容象形文字
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  # 兼容扩展
        return True
    return False

# top-k与top-p筛选逻辑
# 根据概率分布筛选出最可能的几个token
# 参考：https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # 确保logits为一维向量
    top_k = min(top_k, logits.size(-1))  # 检查top_k是否合理
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# 生成文本序列
# 利用GPT模型根据上下文生成指定长度的文本
def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(**inputs)  # 通过模型前向传播得到输出
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated.tolist()[0]):
                next_token_logits[id] /= repitition_penalty  # 进行重复惩罚
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')  # 将未知标记的概率设为无穷小
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # 从筛选后的分布中采样
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

# 主函数
# 用于解析命令行参数，加载模型并生成文本
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度，越高越随机')
    parser.add_argument('--topk', default=8, type=int, required=False, help='生成时选取top k的候选项')
    parser.add_argument('--topp', default=0, type=float, required=False, help='生成时选取累积概率top p的候选项')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False, help='模型参数路径')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
    parser.add_argument('--save_path', default='generated/', type=str, required=False, help='生成文件的保存路径')
    parser.add_argument('--articles_per_title', default=5, type=int, required=False, help='每个标题生成多少篇文章')
    parser.add_argument('--titles', default='萧炎', type=str, required=False, help='标题列表，用空格分隔')
    parser.add_argument('--titles_file', default='', type=str, required=False, help='标题列表文件，如果此项有值则忽略titles参数')
    parser.add_argument('--no_wordpiece', action='store_true', help='不使用word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位切词')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='重复惩罚系数')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # 根据参数选择词汇切分工具
    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    # 设置显卡
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    length = args.length
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty

    # 获取标题列表
    titles = args.titles.split()
    if args.titles_file:
        with open(args.titles_file, 'r') as f:
            titles = [line.strip() for line in f.readlines()]
    articles_per_title = args.articles_per_title
    save_path = args.save_path

    # 选择计算设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载分词器和模型
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    # 创建保存路径
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if length == -1:
        length = model.config.n_ctx

    # 生成文章
    for i, title in enumerate(titles):
        for j in range(articles_per_title):
            with open(f"{save_path}{i}-{j}.txt", 'w') as f:
                context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
                generated = 0
                out = sample_sequence(
                    n_ctx=n_ctx,
                    model=model, length=length,
                    context=context_tokens, tokenizer=tokenizer,
                    temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty,
                    device=device
                )
                out = out.tolist()[0]

                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)

                # 确保英文单词之间有空格
                for i, item in enumerate(text[:-1]):
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '

                # 去除特殊标记
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    if item == '[CLS]' or item == '[SEP]':
                        text[i] = '\n'

                print("=" * 40 + f" SAMPLE {generated} " + "=" * 40)
                text = ''.join(text).replace('##', '').strip()
                print(text)
                f.write(text + '\n')
                print("=" * 80)

if __name__ == '__main__':
    main()
