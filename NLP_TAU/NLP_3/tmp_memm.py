from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
from sklearn.externals import joblib


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features["curr_word"] = curr_word
    ### YOUR CODE HERE
    features["next_word"] = next_word
    features["prev_word"] = prev_word
    features["prevprev_word"] = prevprev_word
    features["prev_tag"] = prev_tag


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
        Returns: feature vector

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


def memm_greedy(sent, logreg, vec):
    sent_copy = [(t[0], None) for t in sent]

    for i in xrange(len(sent_copy)):
        features = extract_features(sent_copy, i)
        fv = vec.transform(features)
        predicted_label = logreg.predict(fv)[0]
        predicted_tag = index_to_tag_dict[predicted_label]
        sent_copy[i] = (sent_copy[i][0], predicted_tag)

    return sent_copy


def transform_feature_dictionary(fv_dict):
    feature_dictionary = {}
    for item in fv_dict.items():
        new_key = "%s_%s" % item
        feature_dictionary[new_key] = 1
    return feature_dictionary


def memm_viterbi(sent, logreg, vec):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    sent_copy = [[t[0], 'None'] for t in sent]
    features = extract_features(sent_copy, 0)
    # print features
    fv = vec.transform(transform_feature_dictionary(features))
    m = len(sent_copy)
    k = len(tagset)
    ### YOUR CODE HERE
    initial_probs = logreg.predict_proba(fv)[0]
    pi = {(0, index_to_tag_dict[i]): initial_probs[i] for i in range(len(tagset))}
    bp = {}
    cache = {}

    for j in range(1, m):
        features = extract_features(sent_copy, j)
        for s in range(k):
            value = float("-inf")
            arg = None
            for s_tag in range(k):
                features['prev_tag'] = index_to_tag_dict[s_tag]
                fv = vec.transform(transform_feature_dictionary(features))

                if (j, s_tag) not in cache:
                    cache[(j, s_tag)] = logreg.predict_proba(fv)[0]

                prob = cache[(j, s_tag)][s]
                candidate = pi[(j - 1, index_to_tag_dict[s_tag])] * prob

                if candidate > value:
                    value = candidate
                    arg = index_to_tag_dict[s_tag]
            pi[(j, index_to_tag_dict[s])] = value
            bp[(j, index_to_tag_dict[s])] = arg

    value = float("-inf")
    for s in range(k):
        candidate = pi[(m - 1, index_to_tag_dict[s])]
        if candidate > value:
            value = candidate
            sent_copy[m - 1][1] = index_to_tag_dict[s]

    for i in range(m - 2, -1, -1):
        sent_copy[i][1] = bp[(i+1, sent_copy[i + 1][1])]

    ### END YOUR CODE
    return sent_copy

def memm_eval(test_data, test_data_vectorized, logreg, vec):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm & greedy hmm
    """
    greedy_correct = 0.0
    viterbi_correct = 0.0
    total = 0.0
    ### YOUR CODE HERE
    for doc in test_data:
        prediction_greedy = memm_greedy(doc, logreg, vec)
        prediction_viterbi = memm_viterbi(doc, logreg, vec)
        for greedy, viterbi, token in zip(prediction_greedy, prediction_viterbi, doc):
            if greedy == token:
                greedy_correct += 1
            if viterbi == token:
                viterbi_correct += 1
            total += 1
    ### END YOUR CODE
    return viterbi_correct / total, greedy_correct / total


if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    # The log-linear model training.
    # NOTE: this part of the code is just a suggestion! You can change it as you wish!
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

    if os.path.exists("logreg_good_10k.pkl"):
        logreg = joblib.load("logreg_good_10k.pkl")
        print "loading model"
    else:
        logreg = linear_model.LogisticRegression(
            multi_class='multinomial', max_iter=10000, solver='lbfgs', C=100000, verbose=2)
        print "Fitting..."
        start = time.time()
        logreg.fit(train_examples_vectorized, train_labels)
        end = time.time()
        print "done, " + str(end - start) + " sec"
        joblib.dump(logreg, "logreg_good_10k.pkl")
    #End of log linear model training

    test_sent = [('Ms.', 'NNP'), ('Haag', 'NNP'), ('plays', 'VBZ'), ('initCap', 'NNP'), ('.', '.')]
    print memm_viterbi(test_sent, logreg, vec)

    #acc_viterbi, acc_greedy = memm_eval(dev_sents, dev_examples_vectorized, logreg, vec)
    #print "dev: acc memm greedy: " + str(acc_greedy)
    #print "dev: acc memm viterbi: " + str(acc_viterbi)
    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        acc_viterbi, acc_greedy = memm_eval(test_sents, logreg, vec)
        print "test: acc memmm greedy: " + acc_greedy
        print "test: acc memmm viterbi: " + acc_viterbi