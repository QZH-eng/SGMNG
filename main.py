import argparse
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import gensim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from utils.bleu_score import computerouge
from data_compute.get_graph_data import PrecomputedDataset
from model.decoder import Decoder
from model.encoder import Encoder
from model.seq2seq import Seq2Seq
from utils.bleu_score import computebleu
from BeamSearch import Predict

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', type=int, default=666, help="random_seed")

parser.add_argument('--hid_dim', type=int, default=256, help='hidden size')
parser.add_argument('--num_features', type=int, default=128, help="num_features")
parser.add_argument('--dropout_ratio', type=float, default=0.6, help='dropout ratio')
parser.add_argument('--n_layers', type=int, default=1, help='gru_n_layers')

parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
parser.add_argument("--num_steps", type=int, default=100000, help="max_epoch")
parser.add_argument('--lr', type=float, default=0.00005, help="lr")
parser.add_argument('--beta_min', type=float, default=0.9, help="beta_min")
parser.add_argument('--beta_max', type=float, default=0.999, help="beta_max")
parser.add_argument('--weight_decay', type=float, default=0.0, help="weight_decay")
parser.add_argument('--store_steps', type=int, default=1, help="store_steps")
parser.add_argument('--summary_steps', type=int, default=1, help="summary_steps")
parser.add_argument("--env", type=str, default=None, help="env")
parser.add_argument('--lr_decay_steps', type=int, default=5000, help="lr_decay_steps")


parser.add_argument("--train_data_path", type=str, default='data/train_dataset.jsonl', help="train_data_path")
parser.add_argument("--test_data_path", type=str, default='data/test_dataset.jsonl', help="test_data_path")
parser.add_argument("--val_data_path", type=str, default='data/val_dataset.jsonl', help="val_data_path")
# parser.add_argument("--code_dict_path", type=str, default='dict/code_dict.pkl',
#                    help="word_dict_path")
# parser.add_argument("--code_vec_path", type=str, default='dict/code_vec.pkl', help="word_vec_path")
# parser.add_argument("--name_dict_path", type=str, default='dict/name_dict.pkl',
#                     help="word_dict_path")
# parser.add_argument("--name_vec_path", type=str, default='dict/name_vec.pkl', help="word_vec_path")
parser.add_argument("--word_dict_path", type=str, default='dict/word_dict.pkl',
                    help="word_dict_path")
parser.add_argument("--ref_dict_path", type=str, default='dict/ref_dict.pkl',
                    help="ref_dict_path")
parser.add_argument("--tgt_dict_path", type=str, default='dict/tgt_dict.pkl',
                    help="tgt_dict_path")
parser.add_argument('--node_dict_path', type=str, default='dict/node_dict.pkl', help="node_name_dict_path")
parser.add_argument('--max_sequence_len', type=int, default=200, help="max_sequence_len")

parser.add_argument('--model_save_path', type=str, default='saved_model', help="model_save_path")
parser.add_argument('--load_model', default=False, help="load_model")
parser.add_argument('--store_model_path', default='./ckpt', help="store_model_path")
parser.add_argument('--load_model_path', default=None, help="load_model_path")
parser.add_argument('--isatt', default=True, help="attention")
parser.add_argument('--coverage', default=False, help="coverage")
parser.add_argument('--LAMBDA', default=1, help="coverageLAMBDA")
parser.add_argument('--pointer', default=True, help="pointer")
parser.add_argument('--greedy_search', default=False, help="greedy_search")

args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'# 这里的0 就是主gpu,  1、2的模型和数据由主gpu分发

# f = open(args.code_vec_path, 'rb')
# code_vec = pickle.load(f).cpu()
# f.close()
#
# f = open(args.code_dict_path, 'rb')
# code_dict = pickle.load(f)
# f.close()
#
# f = open(args.name_vec_path, 'rb')
# name_vec = pickle.load(f).cpu()
# f.close()
#
# f = open(args.name_dict_path, 'rb')
# name_dict = pickle.load(f)
# f.close()

# f = open(args.word_dict_path, 'rb')
# word_dict = pickle.load(f)
# f.close()

f = open(args.ref_dict_path, 'rb')
ref_dict = pickle.load(f)
f.close()

f = open(args.tgt_dict_path, 'rb')
tgt_dict = pickle.load(f)
f.close()

f = open(args.node_dict_path, 'rb')
node_dict = pickle.load(f)
f.close()

# ref2tgt = []
# for x in ref_dict.itos.values():
#     idx = tgt_dict.stoi.get(x)
#     if idx is not None:
#         ref2tgt.append(idx)
#     else:
#         ref2tgt.append(tgt_dict.stoi['<UNK>'])


# f = open(args.node_dict_path, 'rb')
# node_dictionary = pickle.load(f)
# f.close()

# node_vec = torch.zeros(len(node_dictionary.stoi), 128).cpu()
# model = gensim.models.KeyedVectors.load_word2vec_format('node2vec.bin', binary=True)
# for value in node_dictionary.stoi.values():
#     if value:
#         try:
#             node_vec[value] = torch.tensor(model.get_vector(str(value)))
#         except Exception:
#             print(value)
#             continue

args.ref_size = ref_dict.len()
args.tgt_size = tgt_dict.len()
args.node_size = node_dict.len()


def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
    return


def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    return model


def build_model(args):
    # 建構模型
    encoder = Encoder(args).cuda()
    decoder = Decoder(args).cuda()
    model = Seq2Seq(encoder, decoder, ref_dict, tgt_dict).cuda()
    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.99)
    if args.load_model:
        model = load_model(model, args.load_model_path)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)  # 前提是model已经在cuda上了
    return model, optimizer


# def pad_collate(batch):
#     batch_size = len(batch)
#     context_ls = []
#     subtoken_ls = []
#     context_ls.append(batch[0][0])
#     subtoken_ls.append(batch[0][1])
#     node = batch[0][2]
#     # context2subtoken_ls.append(torch.index_select(ref2tgt, 0, batch[0][0]))
#     edge_index = torch.nonzero(batch[0][3]).t()
#     point = len(batch[0][2])
#     graph_indicator = torch.full((1, len(batch[0][2])), 0, dtype=torch.long)
#
#     for i in range(1, batch_size):
#         context_ls.append(batch[i][0])
#         subtoken_ls.append(batch[i][1])
#         node = torch.cat((node, batch[i][2]), dim=0)
#         edge_index = torch.cat((edge_index, torch.nonzero(batch[i][3]).t() + point), dim=-1)
#         graph_indicator = torch.cat((graph_indicator, torch.full((1, len(batch[i][2])), i, dtype=torch.long)), dim=-1)
#         point += len(batch[i][2])
#         # context2subtoken_ls.append(torch.index_select(ref2tgt, 0, batch[i][0]))
#
#     context_ls.extend(subtoken_ls)
#     context_ls = pad_sequence(context_ls, batch_first=True, padding_value=0)
#     # return context_ls[:batch_size, :], context_ls[batch_size: batch_size * 2, :], context_ls[batch_size * 2:, :], context2subtoken_ls
#     return context_ls[:batch_size, :], context_ls[batch_size:, :], node, edge_index, graph_indicator


def pad_collate(batch):
    batch_size = len(batch)
    context_ls = [batch[0]['ref']]
    subtoken_ls = [batch[0]['tgt']]
    len_oov = [batch[0]['len_oov']]
    oov = [batch[0]['oov']]
    ref2tgt = [batch[0]['ref2tgt']]
    node = batch[0]['node']
    map = [batch[0]['map']]
    len_context = [batch[0]['len_context']]
    edge_index = torch.nonzero(batch[0]['adj']).t()
    point = len(batch[0]['node'])
    graph_indicator = torch.full((1, len(batch[0]['node'])), 0, dtype=torch.long)

    for i in range(1, batch_size):
        context_ls.append(batch[i]['ref'])
        subtoken_ls.append(batch[i]['tgt'])
        len_oov.append(batch[i]['len_oov'])
        oov.append(batch[i]['oov'])
        ref2tgt.append(batch[i]['ref2tgt'])
        len_context.append(batch[i]['len_context'])
        node = torch.cat((node, batch[i]['node']), dim=0)
        edge_index = torch.cat((edge_index, torch.nonzero(batch[i]['adj']).t() + point), dim=-1)
        graph_indicator = torch.cat((graph_indicator, torch.full((1, len(batch[i]['node'])), i, dtype=torch.long)), dim=-1)
        map.append(batch[i]['map'] + point)
        point += len(batch[i]['node'])

    context_ls.extend(subtoken_ls)
    context_ls.extend(ref2tgt)
    context_ls = pad_sequence(context_ls, batch_first=True, padding_value=0)
    map = pad_sequence(map, batch_first=True, padding_value=0)
    # return context_ls[:batch_size, :], context_ls[batch_size: batch_size * 2, :], torch.tensor(len_oov, dtype=torch.long), oov, context_ls[batch_size * 2:, :]
    return {'ref': context_ls[:batch_size, :], 'tgt': context_ls[batch_size: batch_size * 2, :], 'len_oov': torch.tensor(len_oov, dtype=torch.long),
            'oov': oov, 'ref2tgt': context_ls[batch_size * 2:, :], 'node': node, 'edge_index': edge_index, 'graph_indicator': graph_indicator,
            'map': map, 'len_context': torch.tensor(len_context, dtype=torch.long)}


# def pad_collate(batch):
#     batch_size = len(batch)
#     context_ls = []
#     subtoken_ls = []
#     map_ls = []
#     len_ls = []
#     edge_type = []
#     node = batch[0][0]
#     edge_index = torch.nonzero(batch[0][1]).t()
#     point = len(batch[0][0])
#     graph_indicator = torch.full((1, len(batch[0][0])), 0, dtype=torch.long)
#     context_ls.append(batch[0][2])
#     len_ls.append(len(batch[0][2]))
#     subtoken_ls.append(batch[0][3])
#     map_ls.append(batch[0][4])
#     for k in range(edge_index.shape[1]):
#         edge_type.append(int(batch[0][1][edge_index[:, k][0]][edge_index[:, k][1]].item()))

#     for i in range(1, batch_size):
#         node = torch.cat((node, batch[i][0]), dim=0)
#         tmp = torch.nonzero(batch[i][1]).t()
#         edge_index = torch.cat((edge_index,  tmp + point), dim=-1)
#         for j in range(tmp.shape[1]):
#             edge_type.append(int(batch[i][1][tmp[:, j][0]][tmp[:, j][1]].item()))
#         map_ls.append(batch[i][4] + point)
#         point += len(batch[i][0])
#         graph_indicator = torch.cat((graph_indicator, torch.full((1, len(batch[i][0])), i, dtype=torch.long)), dim=-1)

#         context_ls.append(batch[i][2])
#         subtoken_ls.append(batch[i][3])
#         len_ls.append(len(batch[i][2]))

#     context_ls.extend(subtoken_ls)
#     context_ls = pad_sequence(context_ls, batch_first=True, padding_value=0)
#     map_ls = pad_sequence(map_ls, batch_first=True, padding_value=0)
#     return node, edge_index, graph_indicator, context_ls[:batch_size, :], context_ls[batch_size:, :], map_ls, torch.tensor(
#         len_ls), torch.tensor(edge_type)


# def pad_collate(batch):
#     batch_size = len(batch)
#     context_ls = []
#     subtoken_ls = []
#     occurance_ls = []
#     context2subtoken_ls = []
#     context_ls.append(batch[0][0])
#     subtoken_ls.append(batch[0][1])
#     occurance_ls.append(batch[0][2])
#     # context2subtoken_ls.append(torch.index_select(ref2tgt, 0, batch[0][0]))

#     for i in range(1, batch_size):
#         context_ls.append(batch[i][0])
#         subtoken_ls.append(batch[i][1])
#         occurance_ls.append(batch[i][2])
#         # context2subtoken_ls.append(torch.index_select(ref2tgt, 0, batch[i][0]))

#     context_ls.extend(subtoken_ls)
#     context_ls.extend(occurance_ls)
#     context_ls = pad_sequence(context_ls, batch_first=True, padding_value=0)
#     # context2subtoken_ls = pad_sequence(context2subtoken_ls, batch_first=True, padding_value=0)
#     # ref target occurance ref2tgt
#     # return context_ls[:batch_size, :], context_ls[batch_size: batch_size * 2, :], context_ls[batch_size * 2:, :], context2subtoken_ls
#     return context_ls[:batch_size, :], context_ls[batch_size: batch_size * 2, :], context_ls[batch_size * 2:, :]

########
# TODO #
########

# 請在這裡直接 return 0 來取消 Teacher Forcing
# 請在這裡實作 schedule_sampling 的策略
def schedule_sampling():
    return 1


def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)


# def tokens2sentence(outputs, int2word):
#     sentences = []
#     for tokens in outputs:
#         sentence = []
#         for token in tokens:
#             word = int2word[int(token)]
#             if word == '<eos>':
#                 break
#             sentence.append(word)
#         sentences.append(sentence)
#
#     return sentences

def outputids2words(outputs, source_oovs, vocab):
    sentences = []
    for i in range(outputs.size(0)):
        sentence = []
        for token in outputs[i]:
            word = vocab.get(int(token), 'UNK')

            if word == '<eos>':
                break

            if word == 'UNK':
                source_oov_idx = token - len(vocab)
                try:
                    word = source_oovs[i][source_oov_idx]
                    sentence.append(word)
                except IndexError:
                    sentence.append('UNK')
            else:
                sentence.append(word)
        sentences.append(sentence)
    return sentences


    # words = []
    # for i in id_list:
    #     try:
    #         w = vocab.index2word[i]  # might be [UNK]
    #     except IndexError:  # w is OOV
    #         assert_msg = "Error: cannot find the ID the in the vocabulary."
    #         assert source_oovs is not None, assert_msg
    #         source_oov_idx = i - vocab.size()
    #         try:
    #             w = source_oovs[source_oov_idx]
    #         except ValueError:  # i doesn't correspond to an source oov
    #             raise ValueError(
    #                 'Error: model produced word ID %i corresponding to source OOV %i \
    #                  but this example only has %i source OOVs'
    #                 % (i, source_oov_idx, len(source_oovs)))
    #     words.append(w)
    # return ' '.join(words)


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0

    # if (total_steps + 1) % args.lr_decay_steps == 0:
    #     for p in optimizer.param_groups:
    #         p['lr'] *= 0.5

    for step in range(summary_steps):
        # node, edge_index, batch, sources, targets, node_maps, lens = next(train_iter)
        # node, edge_index, batch, sources, targets, node_maps, lens = node.cuda(), edge_index.cuda(), batch.cuda(), sources.cuda(), targets.cuda(), node_maps.cuda(), lens.cuda()
        # sources, targets, nodes, edge_index, batches = next(train_iter)
        # sources, targets, nodes, edge_index, batches = sources.cuda(), targets.cuda(), nodes.cuda(), edge_index.cuda(), batches.cuda()
        # sources, targets, len_oovs, oov, ref2tgt = next(train_iter)
        # sources, targets, len_oovs, ref2tgt = sources.cuda(), targets.cuda(), len_oovs.cuda(), ref2tgt.cuda()
        train_data = next(train_iter)
        # outputs, preds = model.forward(node, edge_index, batch, node_vec, sources, targets, node_maps, lens,
        #                                       schedule_sampling(), word_vec)
        # outputs, preds = model(sources, targets, nodes, edge_index, batches, schedule_sampling())
        # outputs, preds, batch_loss = model(sources, targets, len_oovs, schedule_sampling(), ref2tgt)
        outputs, preds, batch_loss = model(train_data, schedule_sampling())
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = train_data['tgt'][:, 1:].reshape(-1)
        # loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        batch_loss.backward()
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        loss_sum += batch_loss.item()
        torch.cuda.empty_cache()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(total_steps + step + 1, loss_sum,
                                                                                   np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses


def test(model, dataloader, loss_function):
    model.eval()
    loss_sum, bleu_score, rouge_score_1, rouge_score_2, rouge_score_l = 0.0, 0.0, 0.0, 0.0, 0.0
    n = 0
    result = []
    # for node, edge_index, batch, sources, targets, node_maps, lens in dataloader:
    #     node, edge_index, batch, sources, targets, node_maps, lens = node.cuda(), edge_index.cuda(), batch.cuda(), sources.cuda(), targets.cuda(), node_maps.cuda(), lens.cuda()
    with torch.no_grad():
        for test_data in dataloader:
            # sources, targets, len_oovs, ref2tgt = sources.cuda(), targets.cuda(), len_oovs.cuda(), ref2tgt.cuda()
            # batch_size = sources.size(0)
            sources = test_data['ref']
            targets = test_data['tgt']
            batch_size =sources.size(0)
            oovs = test_data['oov']
            # outputs, preds = model.inference(node, edge_index, batch, node_vec, sources, targets, node_maps, lens,
            #                                         word_vec)
            outputs, preds, batch_loss = model.inference(test_data)
            # targets 的第一個 token 是 <BOS> 所以忽略
            outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
            targets = targets[:, 1:].reshape(-1)

            # loss = loss_function(outputs, targets)
            # loss_sum += loss.item()
            loss_sum += batch_loss.item()
            # 將預測結果轉為文字
            targets = targets.view(sources.size(0), -1)
            # preds = tokens2sentence(preds, word_dict.itos)
            preds = outputids2words(preds, oovs, tgt_dict.itos)
            # sources = tokens2sentence(sources, context_dict.itos)
            # targets = tokens2sentence(targets, word_dict.itos)
            targets = outputids2words(targets, oovs, tgt_dict.itos)

            # for source, pred, target in zip(sources, preds, targets):
            #    result.append((source, pred, target))
            # 計算 Bleu Score
            # bleu_score += computebleu(preds, targets)
            # 計算 Rouge Score
            score_1, score_2, score_l = computerouge(preds, targets)
            rouge_score_1 += score_1
            rouge_score_2 += score_2
            rouge_score_l += score_l

            n += batch_size

    return loss_sum / len(dataloader), bleu_score / n, result, rouge_score_1 / n, rouge_score_2 / n, rouge_score_l / n


def train_process(args):
    # 準備訓練資料
    train_dataset = PrecomputedDataset(args.train_data_path, args)
    val_dataset = PrecomputedDataset(args.val_data_path, args)
    test_dataset = PrecomputedDataset(args.test_data_path, args)
    test_loader = DataLoader(test_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=True,
                             drop_last=True)

    train_loader = DataLoader(train_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料

    val_loader = DataLoader(val_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=True,
                           drop_last=True)
    # 建構模型
    model, optimizer = build_model(args)
    # loss_function = nn.CrossEntropyLoss(ignore_index=0)
    loss_function = nn.NLLLoss(ignore_index=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < args.num_steps):
        # 訓練模型
        model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, args.summary_steps,
                                       train_dataset)
        train_losses += loss
        # 檢驗模型
        val_loss, bleu_score, result, r_1, r_2, r_l = test(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        total_steps += args.summary_steps
        print("\r",
              "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}, r_1 score: {:.3f} , r_2 score: {:.3f} , r_l score: {:.3f}       ".format(
                  total_steps, val_loss,
                  np.exp(val_loss),
                  bleu_score, r_1, r_2, r_l))
        # 儲存模型和結果
        if total_steps % args.store_steps == 0 or total_steps >= args.num_steps:
            save_model(model, optimizer, args.store_model_path, total_steps)
            # with open(f'{args.store_model_path}/output_{total_steps}.txt', 'w') as f:
            #     for line in result:
            #         print(line, file=f)
            with open(f'{args.store_model_path}/rouge_{total_steps}.txt', 'w') as f:
                print(r_1, r_2, r_l, file=f)

    return train_losses, val_losses, bleu_scores


if __name__ == '__main__':
    print('config:\n', vars(args))
    train_losses, val_losses, bleu_scores = train_process(args)
    model, optimizer = build_model(args)
    pred = Predict(model, ref_dict, tgt_dict)
    test_dataset = PrecomputedDataset(args.test_data_path, args)
    test_loader = DataLoader(test_dataset, collate_fn=pad_collate, batch_size=args.batch_size, shuffle=True,
                             drop_last=True)
    for test_data in test_loader:
        pred.predict(test_data)
