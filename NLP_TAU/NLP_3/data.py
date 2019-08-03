import os
import re
MIN_FREQ = 3


def num_to_word(word):
    if len(word) == 2 and word.isdigit():
        return 'twoDigitNum'
    elif len(word) == 4 and word.isdigit():
        return 'fourDigitNum'
    elif re.search('^[A-Za-z]',word) is not None :
        return 'containsDigitAndAlpha'
    elif re.search('-',word) is  not None :
        return 'containsDigitAndDash'
    elif re.search('/',word) is not None :
        return 'containsDigitAnSlash'
    elif re.search(',',word) is not None :
        return 'containsDigitAndComma'
    elif re.search('[.]',word) is not None :
        return 'containsDigitAndPeriod'
    else:
        return 'otherNum'

def word_to_word(word):
    if re.search('[^A-Z]',word) is None and len(word) > 1:
        return 'allCaps'
    elif re.search('^[A-Z]',word) is not None  and  re.search('[.]$',word) is not None:
        return 'capPeriod'
    elif re.search('^[A-Z]',word) is not None: #and re.search('[A-Z]',word[1:]) is None:
        return 'initCap'
    elif re.search('[^a-z]',word) is None:
        return 'lowerCase'
    else:
        return 'other'



def invert_dict(d):
    res = {}
    for k, v in d.iteritems():
        res[v] = k
    return res

def read_conll_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1

def compute_vocab_count(sents):
    """
        Takes a corpus and computes all words and the number of times they appear
    """
    vocab = {}
    for sent in sents:
        for token in sent:
            increment_count(vocab, token[0])
    return vocab

def replace_word(word):
    """
        Replaces rare words with ctegories (numbers, dates, etc...)
    """
    ### YOUR CODE HERE
    if re.search('[0-9]$',word) is not None:
        return num_to_word(word)
    elif re.search('[a-zA-Z]',word) is not None:
        return word_to_word(word)
    else:
        return 'other'
    ### END YOUR CODE
    return "UNK"

def preprocess_sent(vocab, sents):
    """
        return a sentence, where every word that is not frequent enough is replaced
    """
    res = []
    total, replaced = 0, 0
    for sent in sents:
        new_sent = []
        for token in sent:
            if token[0] in vocab and vocab[token[0]] >= MIN_FREQ:
                new_sent.append(token)
            else:
                new_sent.append((replace_word(token[0]), token[1]))
                replaced += 1
            total += 1
        res.append(new_sent)
    print "replaced: " + str(float(replaced)/total)
    return res


