import torch
import time
import heapq

# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 2000
eps = 1e-20


def source2ids(source_words, vocab, max_sequence_len):
    ids = []
    oovs = []
    unk_id = vocab.stoi['UNK']
    num = 0
    for w in source_words:
        if num < max_sequence_len:
            i = vocab.stoi.get(w, unk_id)
            if i == unk_id:  # If w is OOV
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                # This is 0 for the first source OOV, 1 for the second source OOV
                oov_num = oovs.index(w)
                # This is e.g. 20000 for the first source OOV, 50001 for the second
                ids.append(vocab.len() + oov_num)
            else:
                ids.append(i)
        num += 1
    return ids, oovs


def abstract2ids(abstract_words, vocab, source_oovs):
    """Map tokens in the abstract (reference) to ids.
    """
    ids = []
    unk_id = vocab.stoi['UNK']
    for w in abstract_words:
        i = vocab.stoi.get(w, unk_id)
        if i == unk_id:  # If w is an OOV word
            if w in source_oovs:  # If w is an in-source OOV
                # Map to its temporary source OOV number
                vocab_idx = vocab.len() + source_oovs.index(w)
                ids.append(vocab_idx)
            else:  # If w is an out-of-source OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def replace_oovs(in_tensor, vocab):
    """Replace oov tokens in a tensor with the <UNK> token.
    """
    oov_token = torch.full(in_tensor.shape, vocab.stoi['UNK']).long().cuda()
    out_tensor = torch.where(in_tensor > vocab.len() - 1, oov_token, in_tensor)
    return out_tensor


def timer(module):
    """Decorator function for a timer.

    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        """Wrapper of the timer function.

        Args:
            func (function): The function to be timed.
        """
        def cal_time(*args, **kwargs):
            """The timer function.

            Returns:
                res (any): The returned value of the function being timed.
            """
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper


class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        """
        len_Y = len(self.tokens)
        # Lenth normalization
        ln = (5+len_Y)**alpha / (5+1)**alpha
        cn = beta * torch.sum(  # Coverage normalization
            torch.log(
                eps +
                torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).cuda()
                )
            )
        )

        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.
    """
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)
