import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gap


# class Encoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.dropout_ratio = args.dropout_ratio
#         self.num_features = args.num_features
#         self.hid_dim = args.hid_dim
#         self.batch_size = args.batch_size
#         self.n_layers = args.n_layers

#         # self.conv1 = GatedGraphConv(self.hid_dim, self.n_layers)
#         self.rnn = nn.GRU(self.num_features, self.hid_dim, self.n_layers, batch_first=True,
#                           bidirectional=True)
#         self.dropout = nn.Dropout(self.dropout_ratio)
#         # self.fc = nn.Linear(self.hid_dim, self.hid_dim * 2)


#     def forward(self, x, edge_index, graph_indicator, node_vec, context_ls, context_embedding, node_map, len):
#         # x = torch.index_select(node_vec, 0, x.cpu())
#         # x = x.cuda()
#         # tmp = x
#         # batch = graph_indicator.view(-1)
#         context_vec = []
#         for i in range(context_ls.shape[0]):
#             context_vec.append(torch.index_select(context_embedding, 0, context_ls[i].cpu()))
#         context_vec = torch.stack(context_vec, dim=0).cuda()
#         # context_vec = [batch size, sequence len, embedding dim]
#         outputs, context_vec = self.rnn(self.dropout(context_vec))
#         # outputs = [batch size, sequence len, hid dim * directions]
#         # hidden =  [num_layers * directions, batch size  , hid dim]
#         # outputs 是最上層RNN的輸出
#         # x = self.conv1(x, edge_index)
#         # node_ls = torch.zeros((outputs.shape[0], outputs.shape[1], self.hid_dim))
#         # for i in range(context_ls.shape[0]):
#         #     node_ls[i][:len[i].item()] = torch.index_select(x, 0, node_map[i][:len[i].item()])
#         # x = gmp(x, batch)
#         # x = torch.mul(torch.sigmoid(self.fc1(torch.cat((x, tmp), dim=-1))), self.fc2(x))
#         # x = gap(x, batch)
#         # x = gmp(x, batch)
#         # x = x.expand(context_vec.shape)
#         # context_vec = x + context_vec
#         # node_ls = self.fc(node_ls.cuda())
#         # outputs = node_ls + outputs

#         return outputs, context_vec

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout_ratio = args.dropout_ratio
        self.num_features = args.num_features
        self.hid_dim = args.hid_dim
        self.batch_size = args.batch_size
        self.n_layers = args.n_layers
        self.node_size = args.node_size
        self.ref_size = args.ref_size
        self.node_embedding = nn.Embedding(self.node_size, self.num_features, padding_idx=0)
        self.embedding = nn.Embedding(self.ref_size, self.num_features, padding_idx=0)
        self.conv1 = GatedGraphConv(self.hid_dim, 4)
        self.rnn = nn.GRU(self.num_features, self.hid_dim, self.n_layers, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.W = nn.Linear(self.hid_dim * 3, self.hid_dim * 2)

    def forward(self, input, train_data):
        # context_vec = [batch size, sequence len, embedding dim]
        outputs, hidden = self.rnn(self.dropout(self.embedding(input)))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # outputs 是最上層RNN的輸出
        nodes = train_data['node'].cuda()
        edge_index = train_data['edge_index'].cuda()
        map = train_data['map'].cuda()
        len_context = train_data['len_context'].cuda()
        nodes = self.dropout(self.node_embedding(nodes))
        nodes = self.conv1(nodes, edge_index)
        selected_nodes = torch.zeros(outputs.shape[0], outputs.shape[1], self.hid_dim).cuda()
        for i in range(outputs.shape[0]):
            selected_node = torch.index_select(nodes, 0, map[i][:len_context[i].item()])
            selected_nodes[i][:selected_node.shape[0]] = selected_node
        # nodes = gmp(nodes, graph_indicators.squeeze(0))
        # nodes = nodes.expand(hidden.shape)
        # hidden += nodes
        outputs = torch.cat((outputs, selected_nodes), dim=-1)
        outputs = self.W(outputs)

        return outputs, hidden
