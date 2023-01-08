import random
import sys
import pathlib
import torch


from utils.convert_utils import Beam, timer, add2heap, replace_oovs

max_dec_steps = 30
beam_size = 3
# Beam search
beam_size: int = 3
alpha = 0.2
beta = 0.2
gamma = 2000


class Predict():
    @timer(module='initalize predicter')
    def __init__(self, model, ref_dict, tgt_dict):
        self.model = model
        self.ref_dict = ref_dict
        self.tgt_dict = tgt_dict
        self.BOSID = self.tgt_dict.stoi['<bos>']
        self.EOSID = self.tgt_dict.stoi['<eos>']
        self.UNKID = self.tgt_dict.stoi['UNK']

    # def greedy_search(self,
    #                   x,
    #                   max_sum_len,
    #                   test_data):
    #
    #     # Get encoder output and states.Call encoder forward propagation
    #     encoder_output, encoder_states = self.model.encoder(
    #         replace_oovs(x, self.vocab), self.model.decoder.embedding)
    #
    #     # Initialize decoder's hidden states with encoder's hidden states.
    #     decoder_states = self.model.reduce_state(encoder_states)
    #
    #     # Initialize decoder's input at time step 0 with the SOS token.
    #     x_t = torch.ones(1) * self.vocab.SOS
    #     x_t = x_t.to(self.DEVICE, dtype=torch.int64)
    #     summary = [self.vocab.SOS]
    #     coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
    #     # Generate hypothesis with maximum decode step.
    #     while int(x_t.item()) != (self.vocab.EOS) \
    #             and len(summary) < max_sum_len:
    #         context_vector, attention_weights, coverage_vector = \
    #             self.model.attention(decoder_states,
    #                                  encoder_output,
    #                                  x_padding_masks,
    #                                  coverage_vector)
    #         p_vocab, decoder_states, p_gen = \
    #             self.model.decoder(x_t.unsqueeze(1),
    #                                decoder_states,
    #                                context_vector)
    #         final_dist = self.model.get_final_distribution(x,
    #                                                        p_gen,
    #                                                        p_vocab,
    #                                                        attention_weights,
    #                                                        torch.max(len_oovs))
    #         # Get next token with maximum probability.
    #         x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
    #         decoder_word_idx = x_t.item()
    #         summary.append(decoder_word_idx)
    #         x_t = replace_oovs(x_t, self.vocab)
    #
    #     return summary

    #     @timer('best k')
    def best_k(self, beam, k, encoder_output, x_padding_masks, x, coverage_vector, len_oov, ref2tgt):

        # use decoder to generate vocab distribution for the next token
        x_t = torch.tensor(beam.tokens[-1])
        x_t = x_t.cuda()

        output, hidden, coverage_vector, attn = self.model.decoder(x_t, beam.decoder_states, encoder_output, x_padding_masks, len_oov,
                                                             coverage_vector, ref2tgt)
        # Get context vector from attention network.
        # context_vector, attention_weights, coverage_vector = \
        #     self.model.attention(beam.decoder_states,
        #                          encoder_output,
        #                          x_padding_masks,
        #                          beam.coverage_vector)

        # Replace the indexes of OOV words with the index of OOV token
        # to prevent index-out-of-bound error in the decoder.

        # p_vocab, decoder_states, p_gen = \
        #     self.model.decoder(replace_oovs(x_t, self.vocab),
        #                        beam.decoder_states,
        #                        context_vector)

        # final_dist = self.model.get_final_distribution(x,
        #                                                p_gen,
        #                                                p_vocab,
        #                                                attention_weights,
        #                                                torch.max(len_oovs))
        # Calculate log probabilities.
        log_probs = torch.log(output.squeeze())
        # EOS token penalty. Follow the definition in
        # https://opennmt.net/OpenNMT/translation/beam_search/.
        log_probs[self.EOSID] *= \
            gamma * x.size()[1] / len(beam.tokens)

        log_probs[self.UNKID] = -float('inf')
        # Get top k tokens and the corresponding logprob.
        topk_probs, topk_idx = torch.topk(log_probs, k)

        # Extend the current hypo with top k tokens, resulting k new hypos.
        best_k = [beam.extend(x,
                              log_probs[x],
                              hidden,
                              coverage_vector) for x in topk_idx.tolist()]

        return best_k

    def beam_search(self, max_sum_len, beam_width, test_data):
        res = []
        for i in range(test_data['ref'].shape[0]):
            # run body_sequence input through encoder. Call encoder forward propagation
            input = test_data['ref'][i].cuda()
            target = test_data['tgt'][i].cuda()
            ref2tgt = test_data['ref2tgt'][i].cuda()
            len_oov = test_data['len_oov'][i].cuda()

            x = input
            data = {'node': test_data['node'][i], 'edge_index': test_data['edge_index'][i], 'len_context': test_data['len_context'][i], 'map': test_data['map'][i]}
            input = replace_oovs(input, self.ref_dict)
            encoder_outputs, hidden = self.model.encoder(input.unsqueeze(0), data)

            coverage_vector = None
            x_padding_masks = torch.ne(input, 0).byte().float()

            # initialize the hypothesis with a class Beam instance.

            init_beam = Beam([self.BOSID],
                             [0],
                             hidden,
                             coverage_vector)

            # get the beam size and create a list for stroing current candidates
            # and a list for completed hypothesis
            k = beam_width
            curr, completed = [init_beam], []

            # use beam search for max_sum_len (maximum length) steps
            for _ in range(max_sum_len):
                # get k best hypothesis when adding a new token
                topk = []
                for beam in curr:
                    # When an EOS token is generated, add the hypo to the completed
                    # list and decrease beam size.
                    if beam.tokens[-1] == self.EOSID:
                        completed.append(beam)
                        k -= 1
                        continue
                    for can in self.best_k(beam,
                                           k,
                                           encoder_outputs,
                                           x_padding_masks,
                                           x,
                                           coverage_vector,
                                           len_oov,
                                           ref2tgt
                                           ):
                        # Using topk as a heap to keep track of top k candidates.
                        # Using the sequence scores of the hypos to campare
                        # and object ids to break ties.
                        add2heap(topk, (can.seq_score(), id(can), can), k)

                curr = [items[2] for items in topk]
                # stop when there are enough completed hypothesis
                if len(completed) == beam_width:
                    break
            # When there are not engouh completed hypotheses,
            # take whatever when have in current best k as the final candidates.
            completed += curr
            # sort the hypothesis by normalized probability and choose the best one
            result = sorted(completed,
                            key=lambda x: x.seq_score(),
                            reverse=True)[0].tokens
            res.append(result)
        return res

    @timer(module='doing prediction')
    def predict(self, data, beam_search=True):
        if beam_search:
            summary = self.beam_search(max_sum_len=max_dec_steps, beam_width=beam_size, test_data=data)
        # else:
        #     summary = self.greedy_search(max_sum_len=max_dec_steps, test_data=data)



if __name__ == "__main__":
    # pred = Predict()
    model = torch.load('../model/pgn')
    model.eval()
    pred = Predict(model)
    # print('vocab_size: ', len(pred.vocab))
    # # Randomly pick a sample in test set to predict.
    # with open(config.test_data_path, 'r') as test:
    #     picked = random.choice(list(test))
    #     source, ref = picked.strip().split('<sep>')
    #
    # print('source: ', source, '\n')
    # greedy_prediction = pred.predict(source.split(), beam_search=False)
    # print('greedy: ', greedy_prediction, '\n')
    # beam_prediction = pred.predict(source.split(), beam_search=True)
    # print('beam: ', beam_prediction, '\n')
    # print('ref: ', ref, '\n')
