from PCFG import PCFG
from collections import defaultdict

def load_sents_to_parse(filename):
    sents = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                sents.append(line)
    return sents


def get_probs(pcfg):
    prob_dict = defaultdict(float)
    key = None
    for lhs, rhs_and_weights in pcfg._rules.iteritems():
        for rhs_and_weight in rhs_and_weights:
            rhs = rhs_and_weight[0]

            if len(rhs) == 2:
                key = (lhs,rhs[0],rhs[1])
            else:
                key = (lhs,rhs[0])
                tmp =  rhs_and_weight[1]
                tmp2 = float(pcfg._sums[lhs])
            prob_dict[key] = rhs_and_weight[1] / float(pcfg._sums[lhs])
    return prob_dict

def get_rel_X(word,pcfg,lhs):
    rel_Xs = []
    for l in lhs:
        y_s = pcfg._rules[l]
        for y in y_s:
            if word in y[0]:
                rel_Xs.append(l)
    if len(rel_Xs) >= 1:
        return rel_Xs
    return None

def get_best_tree(i,j,symbol,bp_i_j_X,sent,all_keys):
    s,X_Y_Z = bp_i_j_X[(i,j,symbol)]
    if symbol =='ROOT':
        return "("+str(symbol) +" "+ get_best_tree(i,j-1,X_Y_Z[1],bp_i_j_X,sent,all_keys) +" ("+X_Y_Z[2]+ " "+sent[j]+"))"
    elif (i,s,X_Y_Z[1]) in all_keys and (s+1,j,X_Y_Z[2])  in all_keys:
        return "("+str(symbol) +" "+ get_best_tree(i,s,X_Y_Z[1],bp_i_j_X,sent,all_keys) +" "\
               +get_best_tree(s+1,j,X_Y_Z[2],bp_i_j_X,sent,all_keys)+")"
    elif (i,s,X_Y_Z[1]) not in all_keys and (s+1,j,X_Y_Z[2])  in all_keys:
        return "("+str(symbol) +" ("+X_Y_Z[1]+ " "+sent[i] +") "+get_best_tree(s+1,j,X_Y_Z[2],bp_i_j_X,sent,all_keys)+")"
    elif  (i,s,X_Y_Z[1]) in all_keys and (s+1,j,X_Y_Z[2]) not in all_keys:
        return "("+str(symbol) +" "+ get_best_tree(i,s,X_Y_Z[1],bp_i_j_X,sent,all_keys) +" ("+X_Y_Z[2]+ " "+sent[j]+"))"
    else:
        return "("+str(symbol)+" ("+X_Y_Z[1]+ " "+sent[i] +") ("+X_Y_Z[2]+ " "+sent[j]+"))"

# def split_sentence(sent):
#     sent = sent.split()
#     if 'chief' in sent:
#         new_sent = []
#         i,N = 0,len(sent)
#         while i < N :
#             if sent[i] == 'chief':
#                 tmp = ' '.join(sent[i:i+3])
#                 new_sent.append(tmp)
#                 i += 3
#                 continue
#             new_sent.append(sent[i])
#             i += 1
#         return new_sent
#     return sent


def cky(pcfg, sent):
    ### YOUR CODE HERE
    pi_i_j_x = defaultdict(float)
    bp_i_j_X = defaultdict(list)
    prob_dict = get_probs(pcfg)
    lhs = pcfg._rules.keys()

    # sent = split_sentence(sent)
    sent = sent.lower().split()
    N = len(sent) -1
    rel_lhs = [is_rellhs  for is_rellhs in lhs if is_rellhs not in ['Verb','Det','Adj','Prep','ROOT']]

    for i in range(N):
        word = sent[i]
        X_s = get_rel_X(word,pcfg,lhs)
        if X_s is None:
            return "FAILED TO PARSE!"

        else:
            for j in range(len(X_s)):
                pi_i_j_x[(i,i,X_s[j])] = prob_dict[(X_s[j],word)]

    X_last = get_rel_X(sent[N],pcfg,lhs)


    for l in range(1,N):
        for i in range(N-l):
            j = i+l
            for X in rel_lhs:
                rhs_and_weights = pcfg._rules[X]
                for rhs_and_weight in rhs_and_weights:
                    rhs = rhs_and_weight[0]
                    if len(rhs) == 1:
                        continue
                    key = (X,rhs[0],rhs[1])
                    q = prob_dict[key]
                    best_ij_X = 0.
                    for s in range(i,j):
                        part_1 = pi_i_j_x.get((i,s,rhs[0]),0)
                        part_2 = pi_i_j_x.get((s+1,j,rhs[1]),0)
                        if part_1 == 0 or part_2 == 0:
                            continue
                        pi_i_j_x[(i,j,X)] = q * part_1 * part_2
                        if pi_i_j_x.get((i,j,X),0) > best_ij_X :
                            bp_i_j_X[(i,j,X)] = [s,(X,rhs[0],rhs[1])]
                            best_ij_X = pi_i_j_x[(i,j,X)]

    bp_i_j_X[(0,len(sent)-1,'ROOT')] = [0,('ROOT','S',X_last[0])]
    if '?' in sent:
        bp_i_j_X[(0,len(sent)-1,'ROOT')] = [0,('ROOT','SQues',X_last[0])]
    if (0,N-1,'S') not in bp_i_j_X and (0,N-1,'SQues') not in bp_i_j_X:
        return "FAILED TO PARSE!"
    return get_best_tree(0,len(sent)-1,'ROOT',bp_i_j_X,sent,bp_i_j_X.keys())

    ### END YOUR CODE




if __name__ == '__main__':
    import sys
    pcfg = PCFG.from_file_assert_cnf(sys.argv[1])
    sents_to_parse = load_sents_to_parse(sys.argv[2])
    for sent in sents_to_parse:
        print (cky(pcfg, sent))
