import torch.nn as nn
import torch
import random
from utils.convert_utils import replace_oovs
from queue import PriorityQueue
import operator
# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         assert encoder.n_layers == decoder.n_layers, \
#             "Encoder and decoder must have equal number of layers!"

#     def forward(self, node, edge_index, batch, node_vec, input, target, node_map, len, teacher_forcing_ratio, word_vec):
#         # input  = [batch size, input len, vocab size]
#         # target = [batch size, target len, vocab size]
#         # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
#         batch_size = target.shape[0]
#         target_len = target.shape[1]
#         vocab_size = self.decoder.subtoken_size

#         # 準備一個儲存空間來儲存輸出
#         outputs = torch.zeros(batch_size, target_len, vocab_size).cuda()
#         # 將輸入放入 Encoder
#         encoder_outputs, hidden = self.encoder(node, edge_index, batch, node_vec, input, word_vec, node_map, len)
#         # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
#         # encoder_outputs 主要是使用在 Attention
#         # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
#         # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
#         hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)

#         # [num_layers, batch size, hid dim * directions]
#         hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

#         x_padding_masks = torch.ne(input, 0).byte().float()

#         # 取的 <BOS> token
#         input = target[:, 0]
#         preds = []
#         for t in range(1, target_len):
#             output, hidden, p_gen = self.decoder(input, hidden, encoder_outputs, word_vec, x_padding_masks, coverage_vector)
#             outputs[:, t] = output
#             # 決定是否用正確答案來做訓練
#             teacher_force = random.random() <= teacher_forcing_ratio
#             # 取出機率最大的單詞
#             top1 = output.argmax(1)
#             # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
#             input = target[:, t] if teacher_force and t < target_len else top1
#             preds.append(top1.unsqueeze(1))
#         preds = torch.cat(preds, 1)
#         return outputs, preds

#     def inference(self, node, edge_index, batch, node_vec, input, target, node_map, len, word_vec):
#         ########
#         # TODO #
#         ########
#         # 在這裡實施 Beam Search
#         # 此函式的 batch size = 1
#         # input  = [batch size, input len, vocab size]
#         # target = [batch size, target len, vocab size]
#         batch_size = input.shape[0]
#         input_len = input.shape[1]  # 取得最大字數
#         vocab_size = self.decoder.subtoken_size

#         # 準備一個儲存空間來儲存輸出
#         outputs = torch.zeros(batch_size, input_len, vocab_size).cuda()
#         # 將輸入放入 Encoder
#         encoder_outputs, hidden = self.encoder(node, edge_index, batch, node_vec, input, word_vec, node_map, len)
#         # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
#         # encoder_outputs 主要是使用在 Attention
#         # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
#         # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
#         hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
#         hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
#         # 取的 <BOS> token
#         input = target[:, 0]
#         preds = []
#         for t in range(1, input_len):
#             output, hidden = self.decoder(input, hidden, encoder_outputs, word_vec)
#             # 將預測結果存起來
#             outputs[:, t] = output
#             # 取出機率最大的單詞
#             top1 = output.argmax(1)
#             input = top1
#             preds.append(top1.unsqueeze(1))
#         preds = torch.cat(preds, 1)
#         return outputs, preds


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, ref_dict, tgt_dict):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ref_dict = ref_dict
        self.tgt_dict = tgt_dict
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, train_data, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        input = train_data['ref'].cuda()
        target = train_data['tgt'].cuda()
        len_oov = train_data['len_oov'].cuda()
        ref2tgt = train_data['ref2tgt'].cuda()
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.tgt_size
        # x = input
        input = replace_oovs(input, self.ref_dict)
        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size + torch.max(len_oov)).cuda()
        # 將輸入放入 Encoder
        # encoder_outputs, hidden = self.encoder(self.decoder.embedding(input), nodes, edge_index, batches)
        encoder_outputs, hidden = self.encoder(input, train_data)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)

        # [num_layers, batch size, hid dim * directions]
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        x_padding_masks = torch.ne(input, 0).byte().float()

        # Initialize coverage vector.
        coverage_vector = None
        if self.decoder.coverage:
            coverage_vector = torch.zeros(input.size()).cuda()

        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        step_losses = []

        for t in range(1, target_len):
            output, hidden, coverage_vector, attn = self.decoder(input, hidden, encoder_outputs, x_padding_masks, len_oov, coverage_vector, ref2tgt)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            # Get the probabilities predict by the model for target tokens.
            target_probs = torch.gather(output, 1, target[:, t].unsqueeze(1))
            target_probs = target_probs.squeeze(1)
            input = replace_oovs(input, self.tgt_dict)
            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(target[:, t], 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = -torch.log(target_probs + torch.tensor(1e-31))

            if self.decoder.coverage:
                # Add coverage loss.
                ct_min = torch.min(attn, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=0)
                loss = loss + self.decoder.LAMBDA * cov_loss

            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)
            preds.append(top1.unsqueeze(1))

        preds = torch.cat(preds, 1)
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(target, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return outputs, preds, batch_loss

    def inference(self, test_data):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        input = test_data['ref'].cuda()
        target = test_data['tgt'].cuda()
        ref2tgt = test_data['ref2tgt'].cuda()
        len_oov = test_data['len_oov'].cuda()

        batch_size = input.shape[0]
        input_len = input.shape[1]  # 取得最大字數
        vocab_size = self.decoder.tgt_size

        # x = input
        input = replace_oovs(input, self.ref_dict)
        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size + torch.max(len_oov)).cuda()
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input, test_data)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)

        x_padding_masks = torch.ne(input, 0).byte().float()

        # Initialize coverage vector.
        coverage_vector = None
        if self.decoder.coverage:
            coverage_vector = torch.zeros(input.size()).cuda()

        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        step_losses = []

        for t in range(1, input_len):
            output, hidden, coverage_vector, attn = self.decoder(input, hidden, encoder_outputs, x_padding_masks, len_oov, coverage_vector, ref2tgt)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            input = replace_oovs(input, self.tgt_dict)
            # Get the probabilities predict by the model for target tokens.
            target_probs = torch.gather(output, 1, target[:, t].unsqueeze(1))
            target_probs = target_probs.squeeze(1)

            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(target[:, t], 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = -torch.log(target_probs + 1e-20)

            if self.decoder.coverage:
                # Add coverage loss.
                ct_min = torch.min(attn, coverage_vector)
                cov_loss = torch.sum(ct_min, dim=0)
                loss = loss + self.decoder.LAMBDA * cov_loss

            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)
            preds.append(top1.unsqueeze(1))

        preds = torch.cat(preds, 1)
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(target, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the target sequence length and mean
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return outputs, preds, batch_loss