from data import *
import collections
import numpy as np
def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    """
    ### YOUR CODE HERE
    tmp = {}
    for s in train_data:
        for k,v in s:
            tmp.setdefault(k,[]).append(v)

    word_to_tag = {}
    for k in tmp.keys():
        word_to_tag[k] = collections.Counter(tmp[k]).most_common(1)[0][0]

    return word_to_tag
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    ### YOUR CODE HERE
    denominator = 0
    numerator = 0
    for s in test_set:
        denominator += len(s)
        numerator += reduce(lambda x, y: x + y, [1 if pred_tags[k] == v else 0 for k,v in s])

    accuracy = numerator/float(denominator)
    return str(accuracy)

    ### END YOUR CODE

if __name__ == "__main__":
    train_sents = read_conll_pos_file("Penn_Treebank/train.gold.conll")
    dev_sents = read_conll_pos_file("Penn_Treebank/dev.gold.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    model = most_frequent_train(train_sents)
    print "dev: most frequent acc: " + most_frequent_eval(dev_sents, model)

    if os.path.exists('Penn_Treebank/test.gold.conll'):
        test_sents = read_conll_pos_file("Penn_Treebank/test.gold.conll")
        test_sents = preprocess_sent(vocab, test_sents)
        print "test: most frequent acc: " + most_frequent_eval(test_sents, model)