#!/usr/local/bin/python

from data_utils import utils as du
import numpy as np
import pandas as pd
import csv

# Load the vocabulary
vocab = pd.read_table("data/lm/vocab.ptb.txt", header=None, sep="\s+",
                     index_col=0, names=['count', 'freq'], )

# Choose how many top words to keep
vocabsize = 2000
num_to_word = dict(enumerate(vocab.index[:vocabsize]))
word_to_num = du.invert_dict(num_to_word)

# Load the training set
docs_train = du.load_dataset('data/lm/ptb-train.txt')
S_train = du.docs_to_indices(docs_train, word_to_num)
docs_dev = du.load_dataset('data/lm/ptb-dev.txt')
S_dev = du.docs_to_indices(docs_dev, word_to_num)

def train_ngrams(dataset):
    """
        Gets an array of arrays of indexes, each one corresponds to a word.
        Returns trigram, bigram, unigram and total counts.
    """
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()
    token_count = 0
    ### YOUR CODE HERE
    unigram_counts[('*')] = unigram_counts.get(('*'),len(dataset))
    bigram_counts[('*','*')] = bigram_counts.get(('*','*'),len(dataset))

    for s in dataset:
        token_count += len(s)
        N = len(s)
        get_first_two_ngram(trigram_counts, bigram_counts, unigram_counts,s)

        for i in range(2,N):
            unigram_counts[(s[i])] = unigram_counts.get((s[i]), 0) + 1
            bigram_counts[(s[i-1],s[i])] = bigram_counts.get((s[i-1],s[i]),0) + 1
            trigram_counts[(s[i-2],s[i-1],s[i])] \
                = trigram_counts.get((s[i-2],s[i-1],s[i]),0) + 1

        unigram_counts[('STOP')] = unigram_counts.get(('STOP'), 0) + 1
        bigram_counts[(s[N-1],'STOP')] = bigram_counts.get((s[N-1],'STOP'),0) + 1
        trigram_counts[(s[N-2],s[N-1],'STOP')] = \
            trigram_counts.get((s[N-2],s[N-1],'STOP'),0) + 1

    #adding 'STOP' symbol to count
    token_count += 1

    ### END YOUR CODE
    return trigram_counts, bigram_counts, unigram_counts, token_count

def get_first_two_ngram(trigram_counts, bigram_counts, unigram_counts,s):
            # manual fitting to the first two words
        unigram_counts[(s[0])] = unigram_counts.get((s[0]), 0) + 1
        bigram_counts[('*',s[0])] = bigram_counts.get(('*',s[0]),0) + 1
        trigram_counts[('*','*',s[0])] = trigram_counts.get(('*','*',s[0]),0) + 1

        unigram_counts[(s[1])] = unigram_counts.get((s[1]), 0) + 1
        bigram_counts[(s[0],s[1])] = bigram_counts.get((s[0],s[1]),0) + 1
        trigram_counts[('*',s[0],s[1])] = \
            trigram_counts.get(('*',s[0],s[1]),0) + 1



def evaluate_ngrams(eval_dataset, trigram_counts, bigram_counts, unigram_counts, train_token_count, lambda1, lambda2):
    """
    Goes over an evaluation dataset and computes the perplexity for it with
    the current counts and a linear interpolation
    """
    perplexity = 0
    ### YOUR CODE HERE

    unigram_probs,bigram_probs,trigram_probs = transform_count_to_prob(unigram_counts,bigram_counts,trigram_counts,train_token_count)
    best_lambda_1 = 0
    best_lambda_2 = 0
    best_lambda_3 = 0
    best_perplexity = np.inf
    M = np.sum([len(s) for s in eval_dataset])
    for lambda1 in np.linspace(0, 1, 12):
        for lambda2 in np.linspace(0, 1, 12):
            lambda3 = 1 - lambda1 - lambda2

            if lambda3 < 0:
                continue

            l = calc_p_S(lambda1,lambda2,lambda3,unigram_probs,bigram_probs,trigram_probs,eval_dataset)
            l /=float(M)
            perplexity = np.exp2(-l)
            #print "The perplexity using "+str(lambda1)+", "+str(lambda2)+", "+str(lambda3)+ " is: " +str(perplexity)
            if perplexity < best_perplexity:
                best_lambda_1 = lambda1
                best_lambda_2 = lambda2
                best_lambda_3 = lambda3
                best_perplexity = perplexity
    print
    print "the best perplexity using "+str(best_lambda_1)+", "+str(best_lambda_2)+", "+str(best_lambda_3) +\
          " is: " +str(best_perplexity)

    perplexity = best_perplexity

    ### END YOUR CODE
    return perplexity

def transform_count_to_prob(unigram_counts,bigram_counts,trigram_counts,train_token_count):
    unigram_probs = {k: v / float(train_token_count) for k, v in unigram_counts.items()}
    bigram_probs = {k: v / float(unigram_counts[k[0]]) for k, v in bigram_counts.items()}
    trigram_probs =  {k: v / float(bigram_counts[k[0:2]]) for k, v in trigram_counts.items()}

    return unigram_probs,bigram_probs,trigram_probs


def calc_p_S(lambda1,lambda2,lambda3,unigram_probs,bigram_probs,trigram_probs,eval_dataset):
    l = 0
    for s in eval_dataset:
                p = (lambda3 * unigram_probs.get((s[0]), 0)+ lambda2 * bigram_probs.get(('*',s[0]),0) + \
                     lambda1 * trigram_probs.get(('*','*',s[0]),0)) * \
                    (lambda3 * unigram_probs.get((s[1]), 0) + lambda2 *  bigram_probs.get((s[0],s[1]),0) + \
                     lambda1 * trigram_probs.get(('*',s[0],s[1]),0) )
                # p = 1



                N = len(s)
                for i in range(2,N):
                    p *= (lambda3 * unigram_probs.get((s[i]), 0) + lambda2 *bigram_probs.get((s[i-1],s[i]),0) + \
                        lambda1 * trigram_probs.get((s[i-2],s[i-1],s[i]),0))

                p *= (lambda3 * unigram_probs.get(('STOP'), 0) + lambda2 *  bigram_probs.get((s[-1],'STOP'),0) + \
                    lambda1 *  trigram_probs.get((s[-2],s[-1],'STOP'),0))

                l += np.log2(p)
    return l


def test_ngram():
    """
    Use this space to test your n-gram implementation.
    """
    #Some examples of functions usage
    trigram_counts, bigram_counts, unigram_counts, token_count = train_ngrams(S_train)
    print "#trigrams: " + str(len(trigram_counts))
    print "#bigrams: " + str(len(bigram_counts))
    print "#unigrams: " + str(len(unigram_counts))
    print "#tokens: " + str(token_count)
    perplexity = evaluate_ngrams(S_dev, trigram_counts, bigram_counts, unigram_counts, token_count, 0.5, 0.4)
    print "#perplexity: " + str(perplexity)
    ### YOUR CODE HERE
    ### END YOUR CODE

if __name__ == "__main__":
    test_ngram()
