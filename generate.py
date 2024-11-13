# 导入必要的库
import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel

# 判断一个单词是否是英文单词
def is_word(word):
    for char in word:
        if char not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True

# 判断字符是否是中文字符
def _is_chinese_char(char):
    """
    判断字符是否为CJK字符（中日韩统一表意文字）
    """
    code_point = ord(char)
    if (
        (0x4E00 <= code_point <= 0x9FFF) or
        (0x3400 <= code_point <= 0x4DBF) or
        (0x20000 <= code_point <= 0x2A6DF) or
        (0x2A700 <= code_point <= 0x2B73F) or
        (0x2B740 <= code_point <= 0x2B81F) or
        (0x2B820 <= code_point <= 0x2CEAF) or
        (0xF900 <= code_point <= 0xFAFF) or
        (0x2F800 <= code_point <= 0x2FA1F)
    ):
        return True
    return False

# 使用 top-k 和/或 top-p 过滤 logits
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    通过 top-k 和/或 nucleus (top-p) 过滤 logits
    """
    assert logits.dim() == 1  # 当前只支持 batch size 为 1
    top_k = min(top_k, logits.size(-1))  # 安全检查
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 删除累积概率超过阈值的 token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# 生成文本序列
def sample_sequence(model, context, length, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repetition_penalty=1.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :]
            
            # 对生成的 token 应用重复惩罚
            for token_id in set(generated.tolist()[0]):
                next_token_logits[token_id] /= repetition_penalty
            
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated.tolist()[0]

# 快速生成文本序列
def fast_sample_sequence(model, context, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu'):
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    if len(context) > 1:
        _, past = model(inputs[:, :-1], None)[:2]
        prev = inputs[:, -1].view(1, -1)
    else:
        past = None
        prev = inputs
    
    generated = context[:]
    with torch.no_grad():
        for _ in trange(length):
            output = model(prev, past=past)
            output, past = output[:2]
            output = output[-1].squeeze(0) / temperature
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generated.append(next_token.item())
            prev = next_token.view(1, 1)
    return generated

# 生成文本的主函数
def generate(n_ctx, model, context, length, tokenizer, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, device='cpu', is_fast_pattern=False):
    if is_fast_pattern:
        return fast_sample_sequence(model, context, length, temperature=temperature, top_k=top_k, top_p=top_p, device=device)
    else:
        return sample_sequence(model, context, length, n_ctx, tokenizer, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, device=device)

# 主程序入口
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='指定使用的设备，例如 "0" 表示第一个 GPU')
    parser.add_argument('--length', default=50, type=int, help='生成文本的长度')
    parser.add_argument('--batch_size', default=1, type=int, help='每批生成样本的数量')
    parser.add_argument('--nsamples', default=1, type=int, help='生成样本的总数量')
    parser.add_argument('--temperature', default=1.0, type=float, help='生成温度，值越高生成的文本随机性越强')
    parser.add_argument('--topk', default=30, type=int, help='top-k 采样')
    parser.add_argument('--topp', default=0.0, type=float, help='top-p 采样（nucleus 采样）')
    parser.add_argument('--model_path', default='model/final_model', type=str, help='模型的路径')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, help='词表的路径')
    parser.add_argument('--prefix', default='你好，世界', type=str, help='生成文本的前缀')
    parser.add_argument('--fast_pattern', action='store_true', help='使用快速生成模式')
    parser.add_argument('--save_samples', action='store_true', help='保存生成的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, help='保存样本的路径')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, help='重复惩罚系数')
    
    args = parser.parse_args()
    print('参数列表:' + args.__repr__())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化分词器和模型
    from tokenizations import tokenization_bert
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    n_ctx = model.config.n_ctx
    length = args.length if args.length != -1 else model.config.n_ctx
    
    if args.save_samples:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(os.path.join(args.save_samples_path, 'samples.txt'), 'w', encoding='utf8')
    
    while True:
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(args.prefix))
        generated_samples = 0
        
        for _ in range(args.nsamples // args.batch_size):
            out = generate(
                n_ctx=n_ctx,
                model=model,
                context=context_tokens,
                length=length,
                is_fast_pattern=args.fast_pattern,
                tokenizer=tokenizer,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                repetition_penalty=args.repetition_penalty,
                device=device
            )
            
            for i in range(args.batch_size):
                generated_samples += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, token in enumerate(text[:-1]):
                    if is_word(token) and is_word(text[i + 1]):
                        text[i] = token + ' '
                text = ''.join(text).replace('##', '').strip()
                
                sample_info = f"{'=' * 40} SAMPLE {generated_samples} {'=' * 40}\n"
                print(sample_info)
                print(text)
                
                if args.save_samples:
                    samples_file.write(sample_info)
                    samples_file.write(text + '\n')
                    samples_file.write('=' * 90 + '\n' * 2)
        
        if generated_samples == args.nsamples:
            if args.save_samples:
                samples_file.close()
            break

if __name__ == '__main__':
    main()