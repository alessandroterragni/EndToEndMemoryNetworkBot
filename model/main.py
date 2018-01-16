import data.data_utilities as data_utilities
import e2eMemoryNetwork as e2eMn

from sklearn import metrics
import numpy as np
import argparse
import tensorflow as tf
import pickle as pkl # object serializer
import sys

# directories variables
DATA_DIR = 'data/original/'
P_DATA_DIR = 'data/processed/'
CKPT_DIR= 'ckpt/'




'''
    run prediction on dataset in batches
    S: vectorized stories array
    Q: vectorized questions array
'''
def batch_predict(model, S, Q, batch_size):
    preds = []
    n = len(S)
    # iterates stories in batches 
    for start in range(0, n, batch_size):
        end = start + batch_size
        s = S[start:end]
        q = Q[start:end]
        # pred is the index of the answer with the greatest probability
        pred = model.predict(s, q)
        preds += list(pred)
    return preds





'''
    data preprocessing using data_utilities

'''
def prepare_data(args, task_id):
    # load candidates
    candidates, candid2idx, idx2candid = data_utilities.load_candidates(candidates_f= DATA_DIR + 'candidates.txt')
    
    # create train, testing and validation structures
    train, test, val = data_utilities.load_dialog_task(data_dir= DATA_DIR, task_id= task_id, candid_dic= candid2idx)
    
    # get metadata
    metadata = data_utilities.build_vocab(train + test + val, candidates, args["memory_size"])

    
    # save candidates, train, test and val to disk in a plk file
    data_ = {
            'candidates' : candidates,
            'train' : train,
            'test' : test,
            'val' : val
            }
    with open(P_DATA_DIR + str(task_id) + '.data.pkl', 'wb') as f:
        pkl.dump(data_, f)

    # save metadata to disk in a plk file
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid
    with open(P_DATA_DIR + str(task_id) + '.metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)


        
        
        
'''
    arg parser: specifying which parameters are needed to run the application
    
'''
def parse_args(args):
    parser = argparse.ArgumentParser(description='Train an End To End Memory Network for a task oriented dialog agent')
    
    # mutually exclusive group
    # add_mutually_exclusive_group will make sure that only one of the arguments in the mutually exclusive group was present on the command line
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument('--infer', action='store_true', help='perform inference in an interactive session')
    group.add_argument('--train', action='store_true', help='train model')
    group.add_argument('--prep_data', action='store_true', help='prepare data')
    group.add_argument('--ui', action='store_true',
                        help='interact through web app(flask); do not call this from cmd line')
    
    # optional parameters
    parser.add_argument('--task_id', required=False, type=int, default=1,  
                        help='Task Id in bAbI (6) tasks {1-5}, default = 1, Only task 1 has been implemeented!')
    parser.add_argument('--batch_size', required=False, type=int, default=16,
                        help='batch size, default: 16')
    parser.add_argument('--epochs', required=False, type=int, default=200,
                        help='num iteration of training over train set, default: 200')
    parser.add_argument('--eval_interval', required=False, type=int, default=5,
                        help='num of epochs to evaluate the results, default: 5')
    parser.add_argument('--memory_size', required=False, type=int, default=50,
                        help='memory size for the context, default: 50')
    
    args = vars(parser.parse_args(args))
    return args




# this class is used to perform the actual conversation
class InteractiveSession():
    # initializator
    def __init__(self, model, idx2candid, w2idx, n_cand, memory_size, batch_size):
        self.context = []
        self.u = None
        self.r = None
        self.model = model
        self.idx2candid = idx2candid
        self.w2idx = w2idx
        self.n_cand = n_cand
        self.memory_size = memory_size
        self.batch_size = batch_size
        
    # takes a message and outputs an answer
    def reply(self, msg):
        # removing beginning and end spaces,putting in lower case the message
        line = msg.strip().lower()
        
        # if the message = "exit", end the conversation
        if line == 'exit':
            return('Goodbye ! Hope to see you soon')
        
        # if the message = "clear", we clear the context
        if line == 'clear':
            self.context = []
            reply_msg = 'memory cleared! Restart the conversation from the beginning'
        
        else:
            # tokenizing the message
            u = data_utilities.tokenize(line)
            # creating the data structure ( context, question, answer = -1)
            # at the beginning the context is empty, and the answer is = -1 because it needs to be computed
            data = [(self.context, u, -1)]
            # vectorizing data with the vocabulary
            s, q, a = data_utilities.vectorize_data(data, 
                    self.w2idx, 
                    self.model._sentence_size, 
                    self.batch_size, 
                    self.n_cand, 
                    self.memory_size)
            
            # predicting the answer
            preds = self.model.predict(s,q)
            r = self.idx2candid[preds[0]]
            reply_msg = r
            
            # tokenizing the answer
            r = data_utilities.tokenize(r)
            
            # updating the context
            self.context.append(u)
            self.context.append(r)
        # returning an answer
        return reply_msg

                   

        
        
def main(args):
    # parsing arguments
    args = parse_args(args)

    # preparing data
    if args['prep_data']:
        print('\n>> Preparing Data\n')
        prepare_data(args, task_id=1)
        print(">> Data preparation complete")
        sys.exit()

        
    # reading data and metadata from pickled files
    with open(P_DATA_DIR + str(args['task_id']) + '.metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    with open(P_DATA_DIR + str(args['task_id']) + '.data.pkl', 'rb') as f:
        data_ = pkl.load(f)

    # reading content of data and metadata
    candidates = data_['candidates']
    candid2idx, idx2candid = metadata['candid2idx'], metadata['idx2candid']

    # get train/test/val data
    train, test, val = data_['train'], data_['test'], data_['val']

    # gathering more information from metadata
    sentence_size = metadata['sentence_size']
    w2idx = metadata['w2idx']
    idx2w = metadata['idx2w']
    memory_size = metadata['memory_size']
    vocab_size = metadata['vocab_size']
    n_cand = metadata['n_cand']
    candidate_sentence_size = metadata['candidate_sentence_size']
    batch_size = args['batch_size']
    

    # vectorizing candidates
    candidates_vec = data_utilities.vectorize_candidates(candidates, w2idx, candidate_sentence_size)

    
    # creating e2eMn model
    model = e2eMn.e2eMemoryNetwork(
                vocab_size= vocab_size, 
                sentence_size= sentence_size, 
                embedding_size= 20, 
                candidates_vec= candidates_vec, 
            )
    
    
    # vectorizing data 
    train, val, test, batches = data_utilities.get_batches(train, val, test, metadata, batch_size)
    
    # training the model
    if args['train']:
        epochs = args['epochs']
        eval_interval = args['eval_interval']
        
        # training and evaluation loop
        print('\n>> Training started!\n')
        best_validation_accuracy = 0
        
        # epochs loop
        for i in range(epochs+1):
            
            # fit the model in batches 
            for start, end in batches:
                s = train['s'][start:end]
                q = train['q'][start:end]
                a = train['a'][start:end]
                model.batch_fit(s, q, a)

            
            # if i is a multiple of the evaluation interval and it is not the first cicle, accuracy is computed
            if i%eval_interval == 0 and i!=0:
                
                #predicting answers for training and evaluation
                train_preds = batch_predict(model, train['s'], train['q'], batch_size=batch_size)
                val_preds = batch_predict(model, val['s'], val['q'], batch_size=batch_size)
                
                # computaion of accuray between predicted answers and real answers
                train_acc = metrics.accuracy_score(np.array(train_preds), train['a'])
                val_acc = metrics.accuracy_score(val_preds, val['a'])
                print('Epoch[{}] : <ACCURACY>\n\ttraining : {} \n\tvalidation : {}'.
                     format(i, train_acc, val_acc))
                
                if(val_acc > best_validation_accuracy):
                    best_validation_accuracy = val_acc
                    # save the best model in the check point directory
                    model.saver.save(model.session, CKPT_DIR + '{}/memn2n_model.ckpt'.format(args['task_id']), global_step=i)
        
        print('>>Training completed')
                
    
    
    #inference
    else: 
        # loading and restoring the trained model from disk
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR + str(args['task_id']) )
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from', ckpt.model_checkpoint_path)
            model.saver.restore(model.session, ckpt.model_checkpoint_path)

        
        # caclulating test accuracy
        test_preds = batch_predict(model, test['s'], test['q'], batch_size=batch_size)
        test_acc = metrics.accuracy_score(np.array(test_preds), test['a'])
        print("Testing accuracy: {}".format(test_acc))


        # creating an interactive session instance
        isess = InteractiveSession(model, idx2candid, w2idx, n_cand, memory_size, batch_size)
        print('>> The system is ready')
        print(">> \nType 'exit' to end the program, type 'clear' to restart the conversation")
        print(">> \n Say hello to start the conversation !")

        
        if args['infer']:
            query = ''
            # until the questions is exit, keep answering
            while query != 'exit':
                query = input('>> ')
                print('>> ' + isess.reply(query))
        
        elif args['ui']:
            return isess

                
if __name__ == '__main__':
    main(sys.argv[1:])
