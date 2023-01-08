import torch
import torch.nn as nn

from model.attention import Attention


# class Decoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.subtoken_size = args.subtoken_size
#         self.hid_dim = args.hid_dim * 2
#         self.n_layers = args.n_layers
#         self.isatt = args.isatt
#         self.coverage = args.coverage
#         self.attention = Attention(args.hid_dim)
#         self.dropout_ratio = args.dropout_ratio
#         # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
#         # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
#         # self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
#         self.num_features = args.num_features
#         self.rnn = nn.GRU(self.num_features, self.hid_dim, self.n_layers, batch_first=True)
#         self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
#         self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
#         self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.subtoken_size)
#         self.dropout = nn.Dropout(self.dropout_ratio)
#         self.fc = nn.Linear(self.hid_dim * 2, self.hid_dim)

#     def forward(self, input, hidden, encoder_outputs, subtoken_embedding, x_padding_masks):
#         # input = [batch size, vocab size]
#         # hidden = [num_layers, batch size, hid dim * directions]
#         # Decoder 只會是單向，所以 directions=1
#         subtoken_vec = []
#         for i in range(input.shape[0]):
#             subtoken_vec.append(torch.index_select(subtoken_embedding, 0, input[i].cpu()))
#         subtoken_vec = torch.stack(subtoken_vec, dim=0).cuda()
#         subtoken_vec = self.dropout(subtoken_vec)
#         # embedded = [batch size, 1, emb dim]
#         # Initialize coverage vector.
#         coverage_vector = None
#         if self.coverage:
#             coverage_vector = torch.zeros(input.size())
#         if self.isatt:
#             attn, coverage_vector = self.attention(encoder_outputs, hidden, coverage_vector)
#             # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
#             attn = attn * x_padding_masks
#             normalization_factor = attn.sum(1, keepdim=True)
#             attn = attn / normalization_factor
#             if self.coverage:
#                 coverage_vector = coverage_vector + attn
#             attn = attn.bmm(encoder_outputs)

#         output, hidden = self.rnn(subtoken_vec, hidden)
#         # output = [batch size, 1, hid dim]
#         # hidden = [num_layers, batch size, hid dim]
#         if self.isatt:
#             output = output.view(-1, self.hid_dim)
#             output = torch.cat([output, attn], dim=-1)
#             output = self.fc(output)

#         # 將 RNN 的輸出轉為每個詞出現的機率
#         output = self.embedding2vocab1(output.squeeze(1))
#         output = self.embedding2vocab2(output)
#         output = self.embedding2vocab3(output)
#         # prediction = [batch size, vocab size]
#         p_gen = None
#         if self.pointer:
#             # Calculate p_gen.
#             x_gen = torch.cat([encoder_outputs, hidden.squeeze(0), subtoken_vec.squeeze(1)], dim=-1)
#             p_gen = torch.sigmoid(self.w_gen(x_gen))
#             """Calculate the final distribution for the model.
#             """
#             batch_size = input.size()[0]
#             # Clip the probabilities.
#             p_gen = torch.clamp(p_gen, 0.001, 0.999)
#             # Get the weighted probabilities.
#             p_vocab_weighted = p_gen * output
#             attention_weighted = (1 - p_gen) * attn
#             # Add the attention weights to the corresponding vocab positions.
#             final_distribution =  p_vocab_weighted.scatter_add_(dim=1, index=input, src=attention_weighted)
#             output = final_distribution

#         return output, hidden, p_gen

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tgt_size = args.tgt_size
        self.hid_dim = args.hid_dim * 2
        self.n_layers = args.n_layers
        self.isatt = args.isatt
        self.coverage = args.coverage
        self.LAMBDA = args.LAMBDA
        self.pointer = args.pointer
        self.greedy_search = args.greedy_search
        self.attention = Attention(args.hid_dim)
        self.dropout_ratio = args.dropout_ratio
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        # self.input_dim = emb_dim + hid_dim * 2 if isatt else emb_dim
        self.num_features = args.num_features
        self.embedding = nn.Embedding(self.tgt_size, self.num_features, padding_idx=0)
        self.rnn = nn.GRU(self.num_features, self.hid_dim, self.n_layers, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.num_features)
        self.embedding2vocab2 = nn.Linear(self.num_features, self.tgt_size)
        # self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.subtoken_size)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.w_gen = nn.Linear(self.hid_dim * 2 + self.num_features, 1)

    def forward(self, input, hidden, encoder_outputs, x_padding_masks, len_oov, coverage_vector, ref2tgt):
        # input = [batch size, vocab size]
        # hidden = [num_layers, batch size, hid dim * directions]
        # Decoder 只會是單向，所以 directions=1
        subtoken_vec = self.embedding(input).unsqueeze(1)
        subtoken_vec = self.dropout(subtoken_vec)
        # embedded = [batch size, 1, emb dim]
        if self.isatt:
            attn, coverage_vector = self.attention(encoder_outputs, hidden, coverage_vector, x_padding_masks)
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
            if self.coverage:
                coverage_vector = coverage_vector + attn.squeeze(1)
            encoder_outputs = torch.bmm(attn, encoder_outputs).transpose(0, 1)
            hidden = self.fc(torch.cat([hidden, encoder_outputs], dim=-1))

        output, hidden = self.rnn(subtoken_vec, hidden)

        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]

        # 將 RNN 的輸出轉為每個詞出現的機率
        output = self.embedding2vocab1(output.squeeze(1))
        output = torch.softmax(self.embedding2vocab2(output), dim=1)
        # output = torch.mm(output, torch.t(self.embedding.weight))
        # output = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        if self.pointer:
            # Calculate p_gen.
            # b 1 hid * 2 + embedding_size
            x_gen = torch.cat([encoder_outputs, hidden, subtoken_vec.transpose(0, 1)], dim=-1).transpose(0, 1)
            # b 1 1 - b 1
            p_gen = torch.sigmoid(self.w_gen(x_gen)).squeeze(-1)
            """Calculate the final distribution for the model.
            """
            batch_size = input.size()[0]
            # Clip the probabilities.
            p_gen = torch.clamp(p_gen, 0.001, 0.999)
            # Get the weighted probabilities.
            p_vocab_weighted = p_gen * output
            attention_weighted = (1 - p_gen) * attn.squeeze(1)
            max_oov = torch.max(len_oov)
            extension = torch.zeros((batch_size, max_oov)).cuda()
            p_vocab_extend = torch.cat([p_vocab_weighted, extension], dim=1)
            # Add the attention weights to the corresponding vocab positions.
            final_distribution = p_vocab_extend.scatter_add_(dim=1, index=ref2tgt, src=attention_weighted)
            # output = final_distribution

        return final_distribution, hidden, coverage_vector, attn.squeeze(1)