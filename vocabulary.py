import re


class Vocab(object):
    """vocabulary (terminal symbols or path names or label(method names))"""

    REDUNDANT_SYMBOL_CHARS = re.compile(r"[_0-9]+")
    METHOD_SUBTOKEN_SEPARATOR = re.compile(r"([a-z]+)([A-Z][a-z]+)|([A-Z][a-z]+)")

    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.itosubtokens = {}
        self.freq = {}

    def append(self, name, index=None, subtokens=None):
        if name not in self.stoi:
            if index is None:
                index = len(self.stoi)
            if self.freq.get(index) is None:
                self.freq[index] = 0
            self.stoi[name] = index
            self.itos[index] = name
            if subtokens is not None:
                self.itosubtokens[index] = subtokens
            self.freq[index] += 1
        else:
            index = self.stoi.get(name)
            self.freq[index] += 1

    def get_freq_list(self):
        freq = self.freq
        freq_list = [0] * self.len()
        for i in range(self.len()):
            freq_list[i] = freq[i]
        return freq_list

    def len(self):
        return len(self.stoi)

    @staticmethod
    def normalize_method_name(method_name):
        return Vocab.REDUNDANT_SYMBOL_CHARS.sub("", method_name)

    @staticmethod
    def get_method_subtokens(method_name):
        return [x.lower() for x in Vocab.METHOD_SUBTOKEN_SEPARATOR.split(method_name) if x is not None and x != '']
