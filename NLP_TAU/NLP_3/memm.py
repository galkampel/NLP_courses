from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import numpy as np
import time
import os
import pickle
from sklearn.externals import joblib
import itertools

def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Rerutns: The word's features.
    """
    features = {}
    features['curr'+curr_word] = 1
    # ### YOUR CODE HERE
    features[('prev'+prev_word,'prev'+prev_tag)] =1
    # features[('prevprev'+prevprev_word,'prevprev'+prevprev_tag)] =1
    # features[('w_i-1 ends with ing','t_i-1=VBG')] = 1 if  'ing' in prev_word and 'VBG' == prev_tag  else 0
    # features[('w_i-1 starts with pre','t_i-1=NN')] = 1 if  'pre' in prev_word and 'NN' == prev_tag  else 0
    # features['prevprev'+prevprev_tag] =1
    # features['prev'+prev_tag] =1
    # features[('prevprev'+prevprev_tag,'prev'+prev_tag)] = 1
    # features[('prev'+prev_word,'curr'+curr_word)] = 1
    # features[('prevprev'+prevprev_word,'prev'+prev_word,'curr'+curr_word)] = 1

    ### YOUR CODE HERE

    ### END YOUR CODE
    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<s>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<s>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Rerutns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents):
    print "building examples"
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in xrange(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tagset[sent[i][1]])
    return examples, labels
    print "done"

def memm_greeedy(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    sent_copy = [(t[0], None) for t in sent]
    ### YOUR CODE HERE
    for i in xrange(len(sent)):
        features = extract_features(sent, i)
        fv = vec.transform(features)
        predicted_label = logreg.predict(fv)[0]
        predicted_tag = index_to_tag_dict[predicted_label]
        predicted_tags[i] = predicted_tag
    # print res
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Rerutns: predicted tags for the sentence
    """
    predicted_tags = [""] * (len(sent))
    ### YOUR CODE HERE
    pi_k_u_v = {}

    pi_k_u_v[(-1,'*','*')] = 1
    N = len(sent)
    if N == 1:
        features = extract_features(sent, 0)
        fv = vec.transform(features)
        predicted_label = logreg.predict(fv)[0]
        predicted_tags[0] = index_to_tag_dict[predicted_label]
        return predicted_tags

    bp_k_u_v = {}

    get_first_two_iterations(pi_k_u_v,sent,logreg,vec)
    T = len(index_to_tag_dict)

    for i in xrange(2, N):
        for index_t in xrange(T):
            for index_u in xrange(T):
                t = index_to_tag_dict[index_t]
                word_t = sent[i-2][0]
                u = index_to_tag_dict[index_u]
                word_u = sent[i-1][0]
                sent[i-2] = (word_t,t)
                sent[i-1] = (word_u,u)

                features = extract_features(sent, i)
                fv = vec.transform(features)
                labels_prob = logreg.predict_proba(fv)
                for index_v in xrange(T):
                    v = index_to_tag_dict[index_v]
                    tmp_pi_k_u_v = pi_k_u_v.get((i-1,t,u),0) * labels_prob[0,index_v]
                    if tmp_pi_k_u_v > pi_k_u_v.setdefault((i,u,v),0):
                        pi_k_u_v[(i,u,v)] = tmp_pi_k_u_v
                        bp_k_u_v[(i,u,v)] = t

    u_v_s = [(key[1],key[2]) for key in bp_k_u_v.keys() if (N-1) == key[0] ]
    best_u_v = 0
    for u,v in u_v_s:
        if best_u_v < bp_k_u_v[(N-1,u,v)]:
            best_u_v = bp_k_u_v[(N-1,u,v)]
            predicted_tags[N-1] = v
            predicted_tags[N-2] = u

    for k in range(N-3,-1,-1):
        t_k_2 = predicted_tags[k+2]
        t_k_1 = predicted_tags[k+1]
        # if (k+2,t_k_1,t_k_2) not  in bp_k_u_v:
        #     predicted_tags[k] = np.random.choice(e_tag_counts.keys())
        #     continue
        predicted_tags[k] = bp_k_u_v[(k+2,t_k_1,t_k_2)]
    ### END YOUR CODE
    return predicted_tags


def get_first_two_iterations(pi_k_u_v,sent,logreg,vec):
    u, t= '*', '*'
    features = extract_features(sent, 0)
    fv = vec.transform(features)
    labels_prob = logreg.predict_proba(fv)
    T = len(index_to_tag_dict)
    for i in xrange(T):
        v = index_to_tag_dict[i]
        pi_k_u_v[(0,u,v)] = pi_k_u_v[(-1,t,u)] * labels_prob[0,i]


    for i in xrange(T):
        word_u = sent[0][0]
        sent[0] = (word_u,index_to_tag_dict[i])
        features = extract_features(sent, 1)
        fv = vec.transform(features)
        labels_prob = logreg.predict_proba(fv)

        for j in xrange(T):
            u = index_to_tag_dict[i]
            v = index_to_tag_dict[j]
            tmp_pi_k_u_v = pi_k_u_v[(0,t,u)] * labels_prob[0,j]
            if tmp_pi_k_u_v > pi_k_u_v.setdefault((1,u,v),0):
                pi_k_u_v[(1,u,v)] = tmp_pi_k_u_v




def memm_eval(test_data, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    ## YOUR CODE HERE


    # numerator_greedy = 0.
    numerator_viterbi = 0.
    denominator = 0.
    error_list = []
    test_data = test_data[:20]
    print "len of test data: ", len(test_data)
    for x_s in test_data:
        x,s = zip(*x_s)
        N = len(s)
        denominator += N
        pred_tags_viterbi = memm_viterbi(x_s,logreg,vec)
        # pred_tags_greedy = memm_greeedy(x_s,logreg,vec)
        # numerator_greedy += reduce(lambda x, y: x + y, [1 if pred_tags_greedy[i] == s[i] else 0 for i in range(N)])
        numerator_viterbi += reduce(lambda x, y: x + y, [1 if pred_tags_viterbi[i] == s[i] else 0 for i in range(N)])
        error_list= itertools.chain(error_list, [(s[i],pred_tags_viterbi[i])   for i in xrange(N) if pred_tags_viterbi[i] != s[i] ])
        # print numerator_viterbi
    print 'acc_viterbi: ', (float(numerator_viterbi) / denominator)
    print error_list
    # output = open('error_list.pkl', 'wb')
    # pickle.dump(error_list,output)
    # output.close()
    # acc_greedy = float(numerator_greedy) / denominator
    acc_viterbi = float(numerator_viterbi) / denominator

    ## END YOUR CODE
    return str(acc_viterbi), str(acc_greedy)

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    #The log-linear model training.
    #NOTE: this part of the code is just a suggestion! You can change it as you wish!
    curr_tag_index = 0
    tagset = {}
    for train_sent in train_sents:
        for token in train_sent:
            tag = token[1]
            if tag not in tagset:
                tagset[tag] = curr_tag_index
                curr_tag_index += 1
    index_to_tag_dict = invert_dict(tagset)
    vec = DictVectorizer()
    print "Create train examples"
    train_examples, train_labels = create_examples(train_sents)
    num_train_examples = len(train_examples)
    print "#example: " + str(num_train_examples)
    print "Done"

    print "Create dev examples"
    dev_examples, dev_labels = create_examples(dev_sents)
    num_dev_examples = len(dev_examples)
    print "#example: " + str(num_dev_examples)
    print "Done"

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print "Vectorize examples"
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print "Done"

    if os.path.exists("logreg.pkl"):
        logreg = joblib.load("logreg.pkl")
        print "loading model"
    else:
        logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
        print "Fitting..."
        start = time.time()
        logreg.fit(train_examples_vectorized, train_labels)

        end = time.time()
        print "done, " + str(end - start) + " sec"
        output = open('logreg.pkl', 'wb')
        pickle.dump(logreg,output)
        output.close()
        print 'save pickle'
        #End of log linear model training

    acc_viterbi, acc_greedy = memm_eval(dev_sents, logreg, vec)
    print "dev: acc memm greedy: " + acc_greedy
    print "dev: acc memm viterbi: " + acc_viterbi
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi