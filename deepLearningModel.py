import tensorflow as tf

class DL_RE(object):
    """
    A Deep Learning Model for Relation Extraction.
    Uses an embedding layer, followed by a Deep Neural Network, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, embedding_matrix, POS_vocab, POS_size, position_vocab, position_size, type_vocab, type_size, num_filters, filter_sizes, neurons, l2_reg_lambda, class_weights):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_POS = tf.placeholder(tf.int32, [None, sequence_length], name="input_POS")
        self.input_distance1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_distance1")
        self.input_distance2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_distance2")
        self.input_type = tf.placeholder(tf.int32, [None, sequence_length], name="input_type")
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        # Keeping track of global step
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # Keeping track of l2 regularization loss
        self.l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.Wembedding = tf.Variable(embedding_matrix, name="Wembedding")
            #self.Wembedding = tf.get_variable("Wembedding", shape=embedding_matrix.shape, initializer=tf.constant_initializer(embedding_matrix))
            embedded_chars = tf.nn.embedding_lookup(self.Wembedding, self.input_x)
            if POS_vocab and POS_size:
                self.Wpos = tf.Variable(tf.random_uniform([POS_vocab, POS_size], -1.0, +1.0), name="Wpos")
                embedded_pos = tf.nn.embedding_lookup(self.Wpos, self.input_POS)
                embedded_chars = tf.concat([embedded_chars, embedded_pos], 2)
            if position_vocab and position_size:
                self.Wd1 = tf.Variable(tf.random_uniform([position_vocab, position_size], -1.0, +1.0), name="Wd1")
                embedded_distance1 = tf.nn.embedding_lookup(self.Wd1, self.input_distance1)
                self.Wd2 = tf.Variable(tf.random_uniform([position_vocab, position_size], -1.0, +1.0), name="Wd2")
                embedded_distance2 = tf.nn.embedding_lookup(self.Wd2, self.input_distance2)
                embedded_chars = tf.concat([embedded_chars, embedded_distance1, embedded_distance2], 2)
            if type_vocab and type_size:
                self.Wtype = tf.Variable(tf.random_uniform([type_vocab, type_size], -1.0, +1.0), name="Wtype")
                embedded_type = tf.nn.embedding_lookup(self.Wtype, self.input_type)
                embedded_chars = tf.concat([embedded_chars, embedded_type], 2)

        '''
        # Entity attention layer
        with tf.name_scope("sentence-attention"):
            idx1 = tf.where(tf.equal(self.distance1, sequence_length-1))
            We1 = tf.expand_dims(tf.gather_nd(embedded_chars, idx1), -1)
            idx2 = tf.where(tf.equal(self.distance2, sequence_length-1))
            We2 = tf.expand_dims(tf.gather_nd(embedded_chars, idx2), -1)
            self.alpha1 = tf.nn.softmax(tf.reshape(tf.matmul(embedded_chars, We1), [-1, sequence_length]), name="alpha1")
            self.alpha2 = tf.nn.softmax(tf.reshape(tf.matmul(embedded_chars, We2), [-1, sequence_length]), name="alpha2")
            #r0 = tf.div(tf.add(self.alpha1, self.alpha2), 2)
            #embedded_chars0 = tf.multiply(r, embedded_chars)
            r = tf.div(tf.add(tf.matrix_diag(self.alpha1), tf.matrix_diag(self.alpha2)), 2)
            embedded_chars = tf.matmul(r, embedded_chars) 
        '''

        # Convolution layer for each filter size and number of filters
        h_outputs = []
        for filter_size in filter_sizes:
            h_output = tf.expand_dims(embedded_chars, -2)
            for l, num_filter in enumerate(num_filters):
                with tf.name_scope("convolutional%s-%s-%s" %((l+1), num_filter, filter_size)):
                    # Convolution Layer
                    h_output = tf.reshape(h_output, [-1, h_output.shape[1].value, h_output.shape[3].value, h_output.shape[2].value])
                    filter_shape = [filter_size, h_output.shape[2].value, h_output.shape[3].value, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(h_output, W, strides=[1, 1, 1, 1], padding="VALID", name="cnn"+str(l+1))
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    h_output = tf.pad(h, [[0,0],[0,filter_size-1],[0,0],[0,0]])
                    #h_outputs.append(h_output) # guardar todos las salidas intermedias
            h_outputs.append(h_output) # guardar ultimas salidas
        # Combine all the convolution features
        h_pool = tf.concat(h_outputs, 3)

        '''
        # Recurrent bidirectional layer
        h_outputs = []
        h_output = embedded_chars
        for l, num_filter in enumerate(num_filters):
            with tf.name_scope("recurrent%s-%s" %((l+1), num_filter)):
                #cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_filter)
                #cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_filter)
                cell_fw = tf.nn.rnn_cell.LSTMCell(num_filter, state_is_tuple=True)
                cell_bw = tf.nn.rnn_cell.LSTMCell(num_filter, state_is_tuple=True)
                #cell_fw = tf.nn.rnn_cell.GRUCell(num_filter)
                #cell_bw = tf.nn.rnn_cell.GRUCell(num_filter)
                #cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout)
                #cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, h_output, dtype = tf.float32, scope="rnn"+str(l+1))#, sequence_length = sequence_length)#separately => tf.nn.bidirectional_rnn
                h_output = tf.concat([output_fw, output_bw], 2)#tf.div(tf.add(output_fw, output_bw), 2)
                #h_outputs.append(h_output) # guardar todos las salidas intermedias
        h_outputs.append(h_output) # guardar ultimas salidas
        h_pool = tf.concat(h_outputs, 2)
        h_pool = tf.expand_dims(h_pool, -2)
        '''
        
        num_filters_total = h_pool.shape[3].value

        # Maximum pooling layer
        with tf.name_scope("max-pool"):
            pooled = tf.nn.max_pool(h_pool, ksize=[1, sequence_length, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pooling")#tf.nn.avg_pool
            #pooled_norm = tf.nn.lrn(pooled, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm')
            h_max_pool_flat = tf.reshape(pooled, [-1, num_filters_total])
        h_pool_flat = h_max_pool_flat

        '''
        # Attention pooling layer
        with tf.name_scope("att-pool"):
            # Attention mechanism
            H = tf.reshape(h_pool,[-1, num_filters_total])
            #W = tf.Variable(tf.truncated_normal([num_filters_total, num_filters_total], stddev=0.1), name="W")
            #b = tf.Variable(tf.truncated_normal([num_filters_total], stddev=0.1), name="b")
            #H = tf.nn.xw_plus_b(H, W, b)
            M = tf.tanh(H, name="tanh")
            # Attention weights in the sentence
            w_a = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="w_a")
            #w = tf.Variable(tf.truncated_normal([num_filters, 1], stddev=0.1), name="w")
            #w_a = tf.concat([w]*len(filter_sizes), 0)
            self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(M, w_a), [-1, 1, sequence_length]), name="alpha")
            # Apply Attention
            H_T = tf.reshape(h_pool, [-1, sequence_length, num_filters_total])
            h_att_pool_flat = tf.tanh(tf.reshape(tf.matmul(self.alpha, H_T), [-1, num_filters_total]))
        h_pool_flat = h_att_pool_flat
        '''

        '''
        # Combine all the pooling features
        h_pool_flat = tf.concat([h_max_pool_flat, h_att_pool_flat], 1)
        num_filters_total = num_filters_total * 2
        '''

        # Add dropout
        with tf.name_scope("dropout0"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout)

        # Fully connected layer
        if not neurons == [0]:
            for n, neuron in enumerate(neurons):
                with tf.name_scope("MLP%s-%s" %((n+1), neuron)):
                    W = tf.Variable(tf.truncated_normal([num_filters_total, neuron], stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[neuron]), name="b")
                    self.l2_loss += tf.nn.l2_loss(W)
                    h_fc = tf.nn.tanh(tf.nn.xw_plus_b(h_drop, W, b, name="tanh"))
                num_filters_total = neuron
                # Add dropout
                with tf.name_scope("dropout%s" %(n+1)):
                    h_drop = tf.nn.dropout(h_fc, self.dropout)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.scores)
            weights = tf.reduce_sum(self.input_y * tf.constant(class_weights, dtype=tf.float32), 1)
            self.loss = tf.reduce_mean(losses * weights) + l2_reg_lambda * self.l2_loss
            #regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
            #l2_loss = tf.contrib.layers.apply_regularization(regularizer, weights_list=tf.trainable_variables())
            #self.loss = tf.reduce_mean(losses) + l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")