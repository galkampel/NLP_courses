from data import *
import numpy as np
def hmm_train(sents):
    """
        sents: list of tagged sentences
        Rerutns: the q-counts and e-counts of the sentences' tags
    """
    total_tokens = 0
    q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts = {}, {}, {}, {}, {}
    ### YOUR CODE HERE
    q_uni_counts[('*')] = q_uni_counts.get(('*'),len(sents))
    q_bi_counts[('*','*')] = q_bi_counts.get(('*','*'),len(sents))

    for x_s in sents:
        x,s = zip(*x_s)
        total_tokens += len(s)
        N = len(s)
        if N <= 1:
            q_uni_counts[(s[0])] = q_uni_counts.get((s[0]), 0) + 1
            q_bi_counts[('*',s[0])] = q_bi_counts.get(('*',s[0]),0) + 1
            q_tri_counts[('*','*',s[0])] = q_tri_counts.get(('*','*',s[0]),0) + 1
            e_word_tag_counts[(x[0],s[0])] = e_word_tag_counts.get((x[0],s[0]),0) + 1
            q_uni_counts[('STOP')] = q_uni_counts.get(('STOP'), 0) + 1
            q_bi_counts[(s[N-1],'STOP')] = q_bi_counts.get((s[N-1],'STOP'),0) + 1
            q_tri_counts[('*',s[N-1],'STOP')] = \
            q_tri_counts.get(('*',s[N-1],'STOP'),0) + 1
            continue
        get_first_two_ngram(q_tri_counts, q_bi_counts, q_uni_counts,s)
        e_word_tag_counts[(x[0],s[0])] = e_word_tag_counts.get((x[0],s[0]),0) + 1
        e_word_tag_counts[(x[1],s[1])] = e_word_tag_counts.get((x[1],s[1]),0) + 1
        for i in range(2,N):
            q_uni_counts[(s[i])] = q_uni_counts.get((s[i]), 0) + 1
            q_bi_counts[(s[i-1],s[i])] = q_bi_counts.get((s[i-1],s[i]),0) + 1
            q_tri_counts[(s[i-2],s[i-1],s[i])] \
                = q_tri_counts.get((s[i-2],s[i-1],s[i]),0) + 1

            e_word_tag_counts[(x[i],s[i])] = e_word_tag_counts.get((x[i],s[i]),0) + 1

        q_uni_counts[('STOP')] = q_uni_counts.get(('STOP'), 0) + 1
        q_bi_counts[(s[N-1],'STOP')] = q_bi_counts.get((s[N-1],'STOP'),0) + 1
        q_tri_counts[(s[N-2],s[N-1],'STOP')] = \
            q_tri_counts.get((s[N-2],s[N-1],'STOP'),0) + 1
    e_tag_counts = q_uni_counts.copy()
    #adding 'STOP' symbol to count
    total_tokens += 1

    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts


def get_first_two_ngram(q_tri_counts, q_bi_counts, q_uni_counts,s):
            # manual fitting to the first two words
        q_uni_counts[(s[0])] = q_uni_counts.get((s[0]), 0) + 1
        q_bi_counts[('*',s[0])] = q_bi_counts.get(('*',s[0]),0) + 1
        q_tri_counts[('*','*',s[0])] = q_tri_counts.get(('*','*',s[0]),0) + 1

        q_uni_counts[(s[1])] = q_uni_counts.get((s[1]), 0) + 1
        q_bi_counts[(s[0],s[1])] = q_bi_counts.get((s[0],s[1]),0) + 1
        q_tri_counts[('*',s[0],s[1])] = \
            q_tri_counts.get(('*',s[0],s[1]),0) + 1

def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts, \
                lambda_1,lambda_2,lambda_3):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    unigram_probs,bigram_probs,trigram_probs,e_probs = \
        transform_count_to_prob(q_uni_counts,q_bi_counts,q_tri_counts,e_word_tag_counts ,e_tag_counts ,total_tokens)

    pi_k_u_v = {}
    tri_keys = trigram_probs.keys()
    N = len(sent)
    if N == 1:
        return predict_one_word(pi_k_u_v,unigram_probs,bigram_probs,trigram_probs,e_probs,sent \
                                ,lambda_1,lambda_2,lambda_3)

    fill_two_iterations(pi_k_u_v,unigram_probs,bigram_probs,trigram_probs,e_probs,sent \
                        ,lambda_1,lambda_2,lambda_3)
    word_tags = {}
    for word_tag in e_word_tag_counts.keys():
        word_tags.setdefault(word_tag[0], []).append(word_tag[1])
    bp_k_u_v = {}

    for i in range(2,N):
        v_s = word_tags[sent[i]]
        u_s = word_tags[sent[i-1]]
        w_s = word_tags[sent[i-2]]
        # v_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[i]]
        # u_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[i-1]]
        # w_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[i-2]]
        for v in v_s:
            for u in u_s:
                for w in w_s:

                    tmp_k_u_v = pi_k_u_v.get((i-1,w,u),0) *(lambda_1*unigram_probs.get(w,0) + lambda_2*bigram_probs.get((w,u),0) \
                        + lambda_3*trigram_probs.get((w,u,v),0)) * (e_probs[(sent[i],v)])
                    k_u_v = (i,u,v)
                    if k_u_v in pi_k_u_v:
                        if tmp_k_u_v >pi_k_u_v[k_u_v] :
                            pi_k_u_v[k_u_v] = tmp_k_u_v
                            bp_k_u_v[k_u_v] = w
                    else:
                        pi_k_u_v[k_u_v] = tmp_k_u_v
                        bp_k_u_v[k_u_v] = w
# N+1 word:
    v = 'STOP'
    u_s =word_tags[sent[N-1]]
    w_s =word_tags[sent[N-2]]
    # u_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[N-1]]
    # w_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[N-2]]
    for u in u_s:
        for w in w_s:
            tmp_k_u_v =pi_k_u_v.get((N-1,w,u),0) *(lambda_1*unigram_probs.get(w,0) + lambda_2*bigram_probs.get((w,u),0) \
                    + lambda_3*trigram_probs.get((w,u,v),0))

            k_u_v = (N,u,v)
            if k_u_v in pi_k_u_v:
                if tmp_k_u_v > pi_k_u_v[k_u_v] :
                    pi_k_u_v[k_u_v] = tmp_k_u_v
                    predicted_tags[N-1] = u
                    predicted_tags[N-2] = w

            else:
                pi_k_u_v[k_u_v] = tmp_k_u_v
                predicted_tags[N-1] = u
                predicted_tags[N-2] = w

    for k in range(N-3,-1,-1):

        t_k_2 = predicted_tags[k+2]
        t_k_1 = predicted_tags[k+1]
        # if (k+2,t_k_1,t_k_2) not  in bp_k_u_v:
        #     predicted_tags[k] = np.random.choice(e_tag_counts.keys())
        #     continue
        predicted_tags[k] = bp_k_u_v[(k+2,t_k_1,t_k_2)]

    ### END YOUR CODE
    return predicted_tags

def transform_count_to_prob(unigram_counts,bigram_counts,trigram_counts , e_word_tag_counts ,e_tag_counts,train_token_count):

    unigram_probs = {k: v / float(train_token_count) for k, v in unigram_counts.items()}
    bigram_probs = {k: v / float(unigram_counts[k[0]]) for k, v in bigram_counts.items()}
    trigram_probs =  {k: v / float(bigram_counts[k[0:2]]) for k, v in trigram_counts.items()}
    e_probs = {k: v / float(e_tag_counts[k[1]]) for k, v in e_word_tag_counts.items()}

    return unigram_probs,bigram_probs,trigram_probs,e_probs


def predict_one_word(pi_k_u_v,unigram_probs,bigram_probs,trigram_probs,e_probs,sent,lambda_1,lambda_2,lambda_3):

    predicted_tags = [""] * (len(sent))

    pi_k_u_v[(-1,'*','*')] = 1
    v_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[0]]
    for v in v_s:
        k_u_v = (0,'*',v)
        pi_k_u_v[k_u_v] = pi_k_u_v[(-1,'*','*')] *(lambda_1*unigram_probs['*'] + lambda_2*bigram_probs[('*','*')] \
                        + lambda_3*trigram_probs.get(('*','*',v),0)) * (e_probs[(sent[0],v)])

    v = 'STOP'

    u_s = v_s
    for u in u_s:
        tmp_k_u_v = pi_k_u_v.get((0,'*',u),0) *(lambda_1*unigram_probs['*'] + lambda_2*bigram_probs.get(('*',u),0) \
                + lambda_3*trigram_probs.get(('*',u,v),0))
        k_u_v = (1,u,v)
        if k_u_v in pi_k_u_v:
            if tmp_k_u_v >pi_k_u_v[k_u_v] :
                pi_k_u_v[k_u_v] = tmp_k_u_v
                predicted_tags[0] = u
        else:
            pi_k_u_v[k_u_v] = tmp_k_u_v
            predicted_tags[0] = u

    return predicted_tags




def fill_two_iterations(pi_k_u_v,unigram_probs,bigram_probs,trigram_probs,e_probs,sent,lambda_1,lambda_2,lambda_3):
    #first 2 iterations: it 1- '*','*' it 2- '*','w'
    pi_k_u_v[(-1,'*','*')] = 1
    v_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[0]]
    for v in v_s:
        if ('*','*',v) in trigram_probs.keys():
            k_u_v = (0,'*',v)
            pi_k_u_v[k_u_v] = pi_k_u_v[(-1,'*','*')] *(lambda_1*unigram_probs['*'] + lambda_2*bigram_probs[('*','*')] \
                            + lambda_3*trigram_probs.get(('*','*',v),0)) * (e_probs[(sent[0],v)])


    v_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[1]]
    u_s = [word_tag[1] for word_tag in e_word_tag_counts.keys() if word_tag[0] == sent[0]]
    for v in v_s:
        for u in u_s:
            tmp_k_u_v = pi_k_u_v.get((0,'*',u),0) *(lambda_1*unigram_probs['*'] + lambda_2*bigram_probs.get(('*',u),0) \
                    + lambda_3*trigram_probs.get(('*',u,v),0))* (e_probs[(sent[1],v)])
            k_u_v = (1,u,v)
            if k_u_v in pi_k_u_v:
                if tmp_k_u_v >pi_k_u_v[k_u_v] :
                    pi_k_u_v[k_u_v] = tmp_k_u_v
            else:
                pi_k_u_v[k_u_v] = tmp_k_u_v





def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    acc_viterbi = 0.0
    ### YOUR CODE HERE
    best_lambda_1,best_lambda_2,best_lambda_3 = 0,0,0
    for lambda1 in np.linspace(0, 1, 12):
        for lambda2 in np.linspace(0, 1, 12):
            lambda3 = 1 - lambda1 - lambda2

            if lambda3 < 0:
                continue


            denominator = 0
            numerator = 0
            for x_s in test_data:
                x, s = zip(*x_s)
                pred_tags = hmm_viterbi(x,total_tokens,q_tri_counts,q_bi_counts,q_uni_counts,e_word_tag_counts, \
                                        e_tag_counts,lambda1,lambda2,lambda3)
                denominator += len(s)
                numerator += reduce(lambda x, y: x + y, [1 if pred_tags[i] == s[i] else 0 for i in range(len(s))])

            tmp_acc = numerator/float(denominator)
            # print tmp_acc

            if tmp_acc > acc_viterbi:
                acc_viterbi = tmp_acc
                best_lambda_1 = lambda1
                best_lambda_2 =lambda2
                best_lambda_3 =lambda3
    # print "best lambdas (1,2,3):",(best_lambda_1,best_lambda_2,best_lambda_3)
    acc_viterbi = str(acc_viterbi)
    ### END YOUR CODE
    return acc_viterbi

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)
    acc_viterbi = hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts,e_tag_counts)
    print "dev: acc hmm viterbi: " + acc_viterbi

    if os.path.exists("Penn_Treebank/test.gold.conll"):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi = hmm_eval(test_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                                           e_word_tag_counts, e_tag_counts)
        print "test: acc hmm viterbi: " + acc_viterbi