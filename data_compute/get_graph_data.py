# import pathlib
# import pickle
#
# import jsonlines
# import networkx as nx
# import numpy as np
# import torch
# from loguru import logger
#
# from vocabulary import Vocab
#
#
# class PrecomputedDataset(torch.utils.data.Dataset):
#     """Defines a Dataset of unsupervised programs stored in jsonlines formats."""
#
#     def __init__(
#             self,
#             path,
#             args,
#             limit_size=-1,
#             preloaded_examples=None,
#     ):
#         super().__init__()
#         full_path = pathlib.Path(path).resolve()
#         if preloaded_examples is not None:
#             logger.debug("Using preloaded examples")
#             self.examples = preloaded_examples
#         else:
#             logger.debug(f"Loading {full_path}")
#             self.examples = []
#             if str(path).endswith('.jsonl'):
#                 with open(path, "r+", encoding="utf8") as f:
#                     for item in jsonlines.Reader(f):
#                         self.examples.append(item)
#             else:
#                 raise NotImplementedError(f"Invalid File mode")
#         if limit_size > 0:
#             self.examples = self.examples[:limit_size]
#             logger.debug(f"Limited size: took first {limit_size} examples")
#
#         with open(args.word_dict_path, 'rb') as f:
#             word_dictionary = pickle.load(f)
#
#         with open(args.node_dict_path, 'rb') as f:
#             node_dictionary = pickle.load(f)
#
#         self.word_dictionary = word_dictionary
#         self.node_dictionary = node_dictionary
#
#         self.max_sequence_len = args.max_sequence_len
#
#     def __len__(self):
#         return len(self.examples)
#
#     def __getitem__(self, idx):
#         ContextGraph = self.examples[idx]['ContextGraph']
#         func_name = self.examples[idx]['FunctionName']
#         context = self.examples[idx]['CodeSequence']
#         # node_map = self.examples[idx]['Map']
#         NodeLabels = ContextGraph['NodeLabels']
#         node_dict = self.node_dictionary
#         word_dict = self.word_dictionary
#         code_graph = nx.DiGraph()
#         code_graph.add_edges_from(ContextGraph['Edges']['Child'], weight=1)
#
#         code_graph.add_edges_from(ContextGraph['Edges']['NextToken'], weight=2)
#
#         code_graph.add_edges_from(ContextGraph['Edges']['LastUse'], weight=3)
#
#         code_graph.add_edges_from(ContextGraph['Edges']['ReturnTo'], weight=4)
#
#         code_graph.add_edges_from(ContextGraph['Edges']['ComputeFrom'], weight=5)
#
#         code_graph.add_edges_from(ContextGraph['Edges']['SubToken'], weight=6)
#
#         adj = np.array(nx.adjacency_matrix(code_graph).todense())
#         node_index = []
#         # map_index = []
#         context_index = []
#
#         for node in code_graph.nodes():
#             idx = node_dict.stoi.get(NodeLabels[node])
#             unk_node = node_dict.stoi['UNK']
#             if idx is not None:
#                 node_index.append(idx)
#             else:
#                 node_index.append(unk_node)
#
#
#         sub_token = [word_dict.stoi['<bos>']]
#         for sub in func_name:
#             idx = word_dict.stoi.get(sub)
#             if idx is not None:
#                 sub_token.append(idx)
#             else:
#                 sub_token.append(word_dict.stoi['UNK'])
#
#         sub_token.append(word_dict.stoi['<eos>'])
#
#         num = 0
#         for name in context:
#             if num < self.max_sequence_len:
#                 idx = word_dict.stoi.get(name)
#                 if idx is not None:
#                     context_index.append(idx)
#                 else:
#                     context_index.append(word_dict.stoi['UNK'])
#                 num += 1
#             else:
#                 break
#
#         # nodes_ls = list(code_graph.nodes)
#         # num = 0
#         # for x in node_map:
#         #     if num < 100:
#         #         map_index.append(nodes_ls.index(str(x)))
#         #         num += 1
#         #     else:
#         #         break
#
#         # return torch.tensor(node_index, dtype=torch.long), torch.Tensor(adj), torch.tensor(context_index,
#         #                                                                                    dtype=torch.long), \
#         #        torch.tensor(sub_token, dtype=torch.long), torch.tensor(map_index, dtype=torch.long)
#         # ref target
#
#         return torch.tensor(context_index, dtype=torch.long), torch.tensor(sub_token, dtype=torch.long), \
#                 torch.tensor(node_index, dtype=torch.long), torch.Tensor(adj)

import pathlib
import pickle

import jsonlines
import networkx as nx
import numpy as np
import torch
from loguru import logger
from utils.convert_utils import abstract2ids,source2ids
from vocabulary import Vocab


class PrecomputedDataset(torch.utils.data.Dataset):
    """Defines a Dataset of unsupervised programs stored in jsonlines formats."""

    def __init__(
            self,
            path,
            args,
            limit_size=-1,
            preloaded_examples=None,
    ):
        super().__init__()
        full_path = pathlib.Path(path).resolve()
        if preloaded_examples is not None:
            logger.debug("Using preloaded examples")
            self.examples = preloaded_examples
        else:
            logger.debug(f"Loading {full_path}")
            self.examples = []
            if str(path).endswith('.jsonl'):
                with open(path, "r+", encoding="utf8") as f:
                    for item in jsonlines.Reader(f):
                        self.examples.append(item)
            else:
                raise NotImplementedError(f"Invalid File mode")
        if limit_size > 0:
            self.examples = self.examples[:limit_size]
            logger.debug(f"Limited size: took first {limit_size} examples")

        with open(args.ref_dict_path, 'rb') as f:
            ref_dictionary = pickle.load(f)

        with open(args.tgt_dict_path, 'rb') as f:
            tgt_dictionary = pickle.load(f)

        with open(args.node_dict_path, 'rb') as f:
            node_dictionary = pickle.load(f)

        self.ref_dictionary = ref_dictionary
        self.tgt_dictionary = tgt_dictionary
        self.node_dictionary = node_dictionary

        self.max_sequence_len = args.max_sequence_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        func_name = self.examples[idx]['FunctionName']
        context = self.examples[idx]['CodeSequence']
        ContextGraph = self.examples[idx]['ContextGraph']
        node_map = self.examples[idx]['Map']
        NodeLabels = ContextGraph['NodeLabels']
        ref_dict = self.ref_dictionary
        tgt_dict = self.tgt_dictionary
        node_dict = self.node_dictionary

        code_graph = nx.DiGraph()
        code_graph.add_edges_from(ContextGraph['Edges']['Child'], weight=1)

        code_graph.add_edges_from(ContextGraph['Edges']['NextToken'], weight=2)

        code_graph.add_edges_from(ContextGraph['Edges']['LastUse'], weight=3)

        code_graph.add_edges_from(ContextGraph['Edges']['ReturnTo'], weight=4)

        code_graph.add_edges_from(ContextGraph['Edges']['ComputeFrom'], weight=5)

        code_graph.add_edges_from(ContextGraph['Edges']['SubToken'], weight=6)

        adj = np.array(nx.adjacency_matrix(code_graph).todense())

        node_index = []
        map_index = []

        for node in code_graph.nodes():
            idx = node_dict.stoi.get(NodeLabels[node])
            unk_node = node_dict.stoi['UNK']
            if idx is not None:
                node_index.append(idx)
            else:
                node_index.append(unk_node)

        context_index, oovs = source2ids(context, ref_dict, self.max_sequence_len)

        sub_token = abstract2ids(func_name, tgt_dict, oovs)

        sub_token = [tgt_dict.stoi['<bos>']] + sub_token + [tgt_dict.stoi['<eos>']]

        ref2tgt = []
        for x in context_index:
            if x < ref_dict.len():
                y = tgt_dict.stoi.get(ref_dict.itos.get(x))
                if y is not None:
                    ref2tgt.append(y)
                else:
                    ref2tgt.append(tgt_dict.stoi['UNK'])
            else:
                ref2tgt.append(tgt_dict.len() + (x - ref_dict.len()))

        nodes_ls = list(code_graph.nodes)

        num = 0
        for z in node_map:
            if num < self.max_sequence_len:
                map_index.append(nodes_ls.index(str(z)))
                num += 1
            else:
                break

        # ref tgt len_oov oov ref2tgt node adj map len_context
        # return torch.tensor(context_index, dtype=torch.long), torch.tensor(sub_token, dtype=torch.long), len(oovs), oovs, \
        #        torch.tensor(ref2tgt, dtype=torch.long), torch.tensor(node_index, dtype=torch.long), torch.Tensor(adj), \
        #        torch.tensor(map_index, dtype=torch.long), len(context)

        # ref tgt len_oov oov ref2tgt node adj map len_context
        return {'ref': torch.tensor(context_index, dtype=torch.long), 'tgt': torch.tensor(sub_token, dtype=torch.long),
                'len_oov': len(oovs), 'oov': oovs, 'ref2tgt': torch.tensor(ref2tgt, dtype=torch.long),
                'node': torch.tensor(node_index, dtype=torch.long),
                'adj': torch.Tensor(adj), 'map': torch.tensor(map_index, dtype=torch.long),
                'len_context': len(context)}


