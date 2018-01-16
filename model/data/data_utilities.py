DATA_SOURCE = 'data/original/candidates.txt'
DATA_DIR = 'original/candidates.txt'


import re
import os

from itertools import chain
from six.moves import range, reduce

import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))


# Return the tokens of a sentence including punctuation
def tokenize(sent):
    sent=sent.lower()
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in STOP_WORDS]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result


# from the candidates files it creates the candidates array and two mirrored dictionaries, candidate:id and id:candidate
def load_candidates(candidates_f=DATA_SOURCE):
    candidates, candid2idx, idx2candid = [], {}, {}
    
    with open(candidates_f) as f:
        for i, line in enumerate(f):
            candid2idx[line.strip().split(' ',1)[1]] = i
            candidates.append(tokenize(line.strip()))
            idx2candid[i] = line.strip().split(' ',1)[1]
    return candidates, candid2idx, idx2candid



#parse dialogues per lines with the candidates dictionary
def parse_dialogs_per_response(lines,candid_dic):
  
    data=[]
    context=[]
    u=None
    r=None
   
    # iterates through the lines of the files
    for line in lines:
        # removing space from the beginning and the end of the string
        line=line.strip()
        
        # if the line is not empty
        if line:
            # splitting the line: 
            # line is the whole text
            _, line = line.split(' ', 1)
            
            # if in the line there is a tab (\t) we have a question and an answer 
            if '\t' in line:
                # q = question    a = answer
                q, a = line.split('\t')
                
                # save the id of the answer ( from the candidates dictionary)
                a_id = candid_dic[a]
                
                # tokenizing q and a to get the single words of the sentences
                q= tokenize(q)
                a = tokenize(a)
                
                # data is a list like this (old context, question, answer id)
                data.append((context[:],q[:],a_id))
                
                # updating the old context with the new context
                context.append(q)
                context.append(a)
                # the context appears like this
                # [['hello'], ['hello', 'what', 'can', 'i', 'help', 'you', 'with', 'today'] ]
            
            # if there is no tab we only have an answer we add to to the context
            else:
                a=tokenize(line)
                context.append(a)
       
        # if the line is empty it means the dialogue is finished, thus we clear the context 
        else:
            context=[]
    return data
    # data is like this (context line1, tokenized question line 1, answer id line 1), 
    #                   (context line2, tokenized question line 2, answer id line 2) etc
    
    

def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a data strucure [(context,tokenized question, answer_id),...] 
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)


def load_dialog_task(data_dir, task_id, candid_dic):
    '''Load the nth task from the files 
    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 7

    # retrieving the files corresponding to the task_id passed as a parameter 
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'task{}-'.format(task_id)
    
    train_file = [f for f in files if s in f and 'trn' in f][0]
    test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    
    # transforming the dialogues in data structures [(context,tokenized question, answer_id),...]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    
    return train_data, test_data, val_data



# metadata creation
def build_vocab(data, candidates, memory_size):
    # create a vocabulary of unique words from the dialogues and candidates
    vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q) for s, q, a in data))
    vocab |= reduce(lambda x,y: x|y, (set(candidate) for candidate in candidates) )
    vocab=sorted(vocab)
    
    # create a dictionary from the vocabulary ( word: id )
    w2idx = dict((c, i + 1) for i, c in enumerate(vocab))
        
    # calculate other useful metadata
    max_story_size = max(map(len, (s for s, _, _ in data)))
    memory_size = min(memory_size, max_story_size)
    
    #max length of a sentence in the context
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    #max length of a snetence in the candidates
    candidate_sentence_size=max(map(len,candidates))
    #max length of a question
    query_size = max(map(len, (q for _, q, _ in data)))

    sentence_size = max(query_size, sentence_size) 

    # vocabulary size (+1 for nil word)
    vocab_size = len(w2idx) + 1 
    
    return {
            'w2idx' : w2idx,
            'idx2w' : vocab,
            'sentence_size' : sentence_size,
            'candidate_sentence_size' : candidate_sentence_size,
            'memory_size' : memory_size,
            'vocab_size' : vocab_size,
            'n_cand' : len(candidates)
            } 



# it creates an array with the vectorized candidates
# candidates are vecrtorized using the id of the words of the vocabulary
# every vector has the same dimension padding them with 0 at the end
# if a word is not in the vocabulary, it0s vectorized with a 0
def vectorize_candidates(candidates, word_idx, sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    
    for i,candidate in enumerate(candidates):
        # calculate how many 0 are needed for the padding 
        lc=max(0,sentence_size-len(candidate))
        
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    
    # returns a tensor matrix like this: C = [ [cand1 vectorized], [ cand2 vectorized], [11,23,45,67,87,54,67]....]
    return tf.constant(C,shape=shape)



def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    
    """
    S = []
    Q = []
    A = []
    
    # sort data in descending order using context length
    data.sort(key=lambda x:len(x[0]),reverse=True)
    
    
    for i, (story, query, answer) in enumerate(data):
        
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        
        ss = []
        
        for i, sentence in enumerate(story, 1):
            # how many zeros are need for padding
            ls = max(0, sentence_size - len(sentence))
            # if the word is in the vocaulary you add the id, else you add 0
            ss.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]
        
        # If a story length < memory_size, the story will be padded with empty memories
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)
        
        # calcolo quanti zero devo aggiungere al vettore della query
        lq = max(0, sentence_size - len(query))
        # vettorizzo query come al solito
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq
        
        # ritorno degli array di numpy
        S.append(np.array(ss))
        Q.append(np.array(q))
        A.append(np.array(answer))
    return S, Q, A
    # S,Q,A are like this :
    # S = [ vectorized sentence1 ], [vectorized sentence2 ], [11,23,45,67,87,54,67] ]
    # Q = [ [ vectorized question1 ], [ vectorized question2], [ 11,23,45,67,87,54,67 ]  ]
    # A = [ [ vectorized answer1 ], [ vectorized answer2], [ 11,23,45,67,87,54,67 ]  ]




def get_batches(train_data, val_data, test_data, metadata, batch_size):
    '''
    input  : train data, valid data, test_data:  ( context, tokenized question, answer_id)
             metadata : {batch_size, w2idx, sentence_size, num_cand, memory_size}
    output : batch indices ([start, end]); train, val split into stories, ques, answers

    '''
    w2idx = metadata['w2idx']  # indexed vocabulary ( word: id )
    sentence_size = metadata['sentence_size'] 
    memory_size = metadata['memory_size']
    n_cand = metadata['n_cand']
    
    # vectorized train_data in 3 vectors : stories, questions, answers
    trainS, trainQ, trainA = vectorize_data(train_data, w2idx, sentence_size, batch_size, n_cand, memory_size)
    # vectorized val_data in 3 vectors : stories, questions, answers
    valS, valQ, valA = vectorize_data(val_data, w2idx, sentence_size, batch_size, n_cand, memory_size)
    # vectorized test_data in 3 vectors : stories, questions, answers
    testS, testQ, testA = vectorize_data(test_data, w2idx, sentence_size, batch_size, n_cand, memory_size)
    
    n_train = len(trainS)
    n_val = len(valS)
    n_test = len(testS)
    print("Training Size",n_train)
    print("Validation Size", n_val)
    print("Test Size", n_test)
    
    
    # create batches, for example: n_train = 10 , batch = 2 ->> batches =  (0, 2), (2, 4), (4, 6), (6, 8)
    batches = list(zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size)))

    # pack the train vectors in a dictionary 
    train = { 's' : trainS, 'q' : trainQ, 'a' : trainA } 
    # pack the validation vectors in a dictionary  
    val =   { 's' : valS, 'q' : valQ, 'a' : valA } 
    # pack the test vectors in a dictionary  
    test =   { 's' : testS, 'q' : testQ, 'a' : testA }
    
    return train, val, test, batches

