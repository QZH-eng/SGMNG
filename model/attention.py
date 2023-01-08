import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hidden_size = hid_dim
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.rand(hid_dim))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        self.wc = nn.Linear(1, 2 * self.hidden_size, bias=False)

    def forward(self, encoder_outputs, decoder_hidden, coverage_vector, x_padding_masks):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim * directions]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        timestep = encoder_outputs.size(1)
        # decoder_hidden = [batch size, sequence len, hid dim * directions]
        decoder_hidden = decoder_hidden[-1].repeat(timestep, 1, 1).transpose(0, 1)
        # encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]

        if coverage_vector is not None:
            coverage_features = self.wc(coverage_vector.unsqueeze(2))
            decoder_hidden += coverage_features

        decoder_hidden = self.score(decoder_hidden, encoder_outputs)
        x_padding_masks = x_padding_masks.data.eq(0)
        decoder_hidden.masked_fill_(x_padding_masks, -1e9)
        return F.softmax(decoder_hidden, dim=1).unsqueeze(1), coverage_vector

    # def score(self, hidden, encoder_outputs):
    #     # [B*T*2H]->[B*T*H]
    #     hidden = F.relu(self.attn(torch.tanh(hidden + encoder_outputs)))
    #     hidden = hidden.transpose(1, 2)  # [B*H*T]
    #     encoder_outputs = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
    #     # [B*1*T]
    #     return torch.bmm(encoder_outputs, hidden).squeeze(1)  # [B*T]

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        hidden = torch.tanh(self.attn(hidden + encoder_outputs))
        hidden = hidden.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        # [B*1*T]
        return torch.bmm(v, hidden).squeeze(1)  # [B*T]
