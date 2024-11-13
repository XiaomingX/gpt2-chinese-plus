import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder

# 生成训练数据集的函数
def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    # 读取原始数据文件
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        # 将\n替换为[SEP]，表示段落的分隔
        lines = [line.replace('\n', ' [SEP] ') for line in lines]
    all_len = len(lines)
    # 如果没有该目录，创建它
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    
    # 将数据分片编码并保存
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 最后一片包含剩余数据
        sublines = [full_tokenizer.tokenize(line) for line in sublines if len(line) > min_length]  # 只考虑长度超过min_length的句子
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 每篇文章的开头添加MASK，表示文章的开始
            full_line.extend(subline)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 每篇文章的最后添加CLS，表示文章的结束
        # 将编码后的数据写入文件
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')

# 主运行函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, help='设置使用哪些显卡')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, help='选择模型参数')
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, help='选择词库')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=5, type=int, help='训练轮数')
    parser.add_argument('--batch_size', default=8, type=int, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=768, type=int, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, help='滑块积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度训练')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, help='fp16选项')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='最大求和法')
    parser.add_argument('--num_pieces', default=100, type=int, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, help='最短收录文章长度')
    parser.add_argument('--output_dir', default='model/', type=str, help='模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, help='预训练模型路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, help='Tensorboard路径')
    parser.add_argument('--segment', action='store_true', help='是否以词为单位')
    parser.add_argument('--bpe_token', action='store_true', help='subword分片编码')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json路径")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe路径")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # 根据用户选择词汇分词器
    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 设置使用的GPU

    # 读取模型配置
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    # 选择分词器（BPE或词汇分词器）
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    # 训练参数设置
    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    # 创建输出目录如果不存在
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 如果还没有tokenize的数据，先执行tokenize
    if raw:
        print('building files')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        print('files built')

    # 创建GPT2模型
    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    # 计算模型参数总数
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    # 配置优化器与应用计划器
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                  t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # 如果可用GPU数较多，使用多卡
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0

    # 训练循环
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // batch_size):  # drop last
                # 准备数据
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = [torch.tensor([int(x) for x in ids]).long().to(device) for ids in batch]
                batch_inputs = torch.stack(batch_inputs)

                # 正向通过
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]

                # 获得loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                # 后向传播loss
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # 进行优化步骤
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                # 每应用一定次数的步骤，汇报loss
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        # 每轮训练完成后保存模型
        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    # 训练完成后保存最终模型
    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')

if __name__ == '__main__':
    main()
