from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score


def computerouge(sentences, targets):
    score_1, score_2, score_l = 0, 0, 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token:
                if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                    tmp.append(token)
                else:
                    tmp += [word for word in token]
        return tmp

    rouge = Rouge()
    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        if sentence and target:
            sentence = ' '.join(sentence)
            target = ' '.join(target)
            scores = rouge.get_scores(sentence, target)
            score_1 += scores[0]['rouge-1']['f']
            score_2 += scores[0]['rouge-2']['f']
            score_l += scores[0]['rouge-l']['f']
    return score_1, score_2, score_l
