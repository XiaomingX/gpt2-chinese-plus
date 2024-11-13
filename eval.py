import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel

def build_files(data_path, tokenized_data_path, num_pieces, tokenizer, min_length):
    """
    将原始数据转换为分片化的日志文件
    
    :param data_path: 原始数据路径
    :param tokenized_data_path: 分片化日志文件路径
    :param num_pieces: 数据分成的片数
    :param tokenizer: 分词器
    :param min_length: 数据最小长度
    :return:
    """
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]
        all_len = len(lines)
    
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])
        sublines = [tokenizer.tokenize(line) for line in sublines if len(line) > min_length]
        sublines = [tokenizer.convert_tokens_to_ids(line) for line in sublines]
        
        full_line = []
        for subline in sublines:
            full_line.append(tokenizer.convert_tokens_to_ids('[MASK]'))
            full_line.extend(subline)
            full_line.append(tokenizer.convert_tokens_to_ids('[CLS]'))
        
        with open(os.path.join(tokenized_data_path, f'tokenized_train_{i}.txt'), 'w') as f:
            f.write(' '.join(map(str, full_line)))
    print('finish')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, help='设置使用的GPU')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, help='选择词典')
    parser.add_argument('--raw_data_path', default='data/eval.json', type=str, help='原始语料路径')
    parser.add_argument('--tokenized_data_path', default='data/tokenized_eval/', type=str, help='分词后语料输出路径')
    parser.add_argument('--raw', action='store_true', help='是否进行分词')
    parser.add_argument('--batch_size', default=8, type=int, help='batch 的大小')
    parser.add_argument('--log_step', default=1, type=int, help='每多少步数输出日志')
    parser.add_argument('--stride', default=768, type=int, help='每次位移的步长')
    parser.add_argument('--num_pieces', default=100, type=int, help='数据分片的数量')
    parser.add_argument('--min_length', default=128, type=int, help='最小长度')
    parser.add_argument('--pretrained_model', default='', type=str, help='预训练的模型路径')
    parser.add_argument('--output_dir', default='eval_result/', type=str, help='结果输出路径')

    args = parser.parse_args()
    print(f'args:\n{args}')

    from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print(f'config:\n{model_config.to_json_string()}')

    n_ctx = model_config.n_ctx
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    tokenizer.max_len = n_ctx
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.raw:
        print('building files')
        build_files(
            data_path=args.raw_data_path,
            tokenized_data_path=args.tokenized_data_path,
            num_pieces=args.num_pieces,
            tokenizer=tokenizer,
            min_length=args.min_length
        )
        print('files built')

    if not args.pretrained_model:
        print('please specify a trained model.')
        exit(1)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    
    model.eval()
    model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f'number of parameters: {num_parameters}')

    multi_gpu = torch.cuda.device_count() > 1
    if multi_gpu:
        print(f'Let\'s use {torch.cuda.device_count()} GPUs!')
        model = DataParallel(model)
    
    print('starting evaluation')
    total_loss, total_steps = 0, 0
    now = datetime.now()
    print(f'time: {now}')

    for i in range(args.num_pieces):
        with open(os.path.join(args.tokenized_data_path, f'tokenized_train_{i}.txt'), 'r') as f:
            tokens = list(map(int, f.read().strip().split()))

        start_point = 0
        samples = []
        while start_point < len(tokens) - n_ctx:
            samples.append(tokens[start_point: start_point + n_ctx])
            start_point += args.stride
        
        if start_point + n_ctx < len(tokens):
            last = tokens[start_point:]
            last += [tokenizer.convert_tokens_to_ids('[PAD]')] * (n_ctx - len(last))
            samples.append(last)
        
        random.shuffle(samples)
        for step in range(len(samples) // args.batch_size):
            batch = samples[step * args.batch_size: (step + 1) * args.batch_size]
            batch_inputs = torch.tensor(batch).long().to(device)
            batch_labels = batch_inputs.clone()

            outputs = model(input_ids=batch_inputs, labels=batch_labels)
            loss = outputs[0]

            if multi_gpu:
                loss = loss.mean()
            total_loss += loss.item()
            total_steps += 1

            if (total_steps % args.log_step) == 0:
                print(f'time: {datetime.now().hour}:{datetime.now().minute}, Step {step + 1} of piece {i}, ppl: {torch.exp(loss)}')

    with open(os.path.join(args.output_dir, 'result.txt'), 'w') as f:
        f.write(str(np.exp(total_loss / total_steps)))

if __name__ == '__main__':
    main()
