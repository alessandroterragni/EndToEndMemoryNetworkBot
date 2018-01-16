import tensorflow as tf

"""End-To-End Memory Network."""
class e2eMemoryNetwork(object):
    
    def __init__(self, vocab_size, sentence_size, embedding_size, candidates_vec):

        self._vocab_size = vocab_size #The size of the vocabulary (should include the nil word, encoded with 0)
        self._sentence_size = sentence_size   #The size of the embedded sentences
        self._embedding_size = embedding_size #The size of the word embedding
        self._hops = 3  # how many layers the memory network will have
        self._candidates = candidates_vec # vector of the candidates
        self.initializer = tf.random_normal_initializer(stddev=0.1) #Weight initializer: it generates tensors with a normal distribution

        self.session= tf.Session()  #Tensorflow Session the model is run with
        self.optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1e-8) #optimizer: Optimizer algorithm used for the stochastic gradient descent
        
        
        # tensorflow place holder (stories, queries and answers ) a placeholder is a promise to provide a value later
        self._stories = tf.placeholder(tf.int32, [None, None, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None], name="answers")
        
        
        # tensorflow variables
        nil_word_slot = tf.zeros([1, self._embedding_size]) # vector of zeros
            
        # matrix A with dimension ( voc_size, embedding size), the first row of the matrix is a vector of all zeros
        # all the others rows are filled with random weights coming from a normal distribution
        A = tf.concat([ nil_word_slot, self.initializer([self._vocab_size-1, self._embedding_size]) ], 0)
        self.A = tf.Variable(A, name="A")
            
        # matrix H with dimension ( embedding size,  embedding size)
        self.H = tf.Variable(self.initializer([self._embedding_size, self._embedding_size]), name="H")
            
        # matrix W with dimension ( voc_size, embedding size)
        W = tf.concat([ nil_word_slot, self.initializer([self._vocab_size-1, self._embedding_size]) ], 0)
        self.W = tf.Variable(W, name="W")
        
        # creation of a set of varibales names who has the first row filled with nil_words
        self._nil_vars = set([self.A.name,self.W.name])

        
        # calculate answers probilities using memory networks formulas
        predictedProbabilities = self._inference(self._stories, self._queries) 
    
        # Computes sparse softmax cross entropy between predictedProbabilities and answers
        # cross_entropy is tensor of the same shape as answers and of the same type as predictedProbabilities with the softmax cross entropy loss.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictedProbabilities, labels=self._answers, name="cross_entropy")
        
        # Define a loss operation computing the mean of elements of the tensor and it returns the reduced tensor
        loss_op = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
        
        
        
        # cross entropy minimization
        #instead of doing it directly with minimize(), we perform 3 different steps beacuse we need to modify the gradients because of the nill varibales
        
        # step 1
        # computing gradients of loss ( cross entropy mean)  
        # It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable".
        # In this case the list will look like this (gradientForA, A), (gradientForW, W), (gradientForH, H), 
        grads_and_vars = self.optimizer.compute_gradients(loss_op)
       
        # step 2: gradients modification
        # the nill_slot should not not be trained 
        # iterating through the variables, if the varibable has the first row filled with zeros: the first row of the corresponding gradient is overwrited with zero as well,        else the gradients is saved normally
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                t = tf.convert_to_tensor(g)
                s = tf.shape(t)[1]
                z = tf.zeros(tf.stack([1, s]))
                p = tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0)

                nil_grads_and_vars.append((p, v))
            else:
                nil_grads_and_vars.append((g, v))
        
        
        # step 3: apply modified gradients
        # it returns an operation to minimize the cross entropy
        train_op = self.optimizer.apply_gradients(nil_grads_and_vars, name="train_op")

        
    
        # predict operation         
        #max: Returns the index with the largest probability across axes of a tensor
        predict_op = tf.argmax(predictedProbabilities, 1, name="predict_op")
        
        
        # assign operations to the class
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.train_op = train_op
        
        # initializing global variables and run the graph
        init_op = tf.global_variables_initializer()    
        self.session.run(init_op)
        
        # saver class in order to save a checkpoint ( max_to_keep = 1 means you retain just one checkpoint, overwriting the old ones )
        self.saver = tf.train.Saver(max_to_keep=1)
    
    
    
    
    # end to end memory network formula 
    def _inference(self, stories, queries):
       
        # queries embdedding
        q_emb = tf.nn.embedding_lookup(self.A, queries)
        # sum the values in the rows to otain the internal state u
        u_0 = tf.reduce_sum(q_emb, 1)
        u_k = u_0
                        
        # iterate trough the single layers of the memory network
        for _ in range(self._hops):
            
            m_emb = tf.nn.embedding_lookup(self.A, stories)
            m = tf.reduce_sum(m_emb, 2) # sum each sentence vector 
                
            # Take the transpose of the matrix u
            u_transpose = tf.transpose(tf.expand_dims(u_k, -1), [0, 2, 1])
                
            # Calculation of probabilities p_i using softmax
            p = tf.nn.softmax(tf.reduce_sum(m * u_transpose, 2))
            
            p_transpose = tf.transpose(tf.expand_dims(p, -1), [0, 2, 1])
            c_transpose = tf.transpose(m, [0, 2, 1])
                
            # weighted sum between c and p
            o_k = tf.reduce_sum(c_transpose * p_transpose, 2)

            # u_(k+1) = H * u_k + o_k    
            u_k = tf.matmul(u_k, self.H) + o_k
                
        # candidates embedding
        candidates_emb = tf.nn.embedding_lookup(self.W, self._candidates)
        candidates_emb_sum = tf.reduce_sum(candidates_emb,1)
            
        # u * W 
        return tf.matmul(u_k,tf.transpose(candidates_emb_sum))
    
    
    
    # train the algorithm in bacthes: it returns the loss
    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        # dictionary to fill the placeholders of the operation
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        
        loss, _ = self.session.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss
        
    
    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        # dictionary to fill the placeholders of the operation
        feed_dict = {self._stories: stories, self._queries: queries}
        
        # returns the index of the answer with the gratest probability
        return self.session.run(self.predict_op, feed_dict=feed_dict)
