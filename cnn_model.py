import tensorflow as tf


class TCNNConfig(object):
    embedding_size = 128 
    num_filters=128
    #filter_sizes="3,4,5"
    l2_reg_lambda=0.0
class TextCNN(object):

    def __init__(self,sequence_length, num_classes, vocab_size,filter_sizes,config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.cnn(sequence_length, num_classes, vocab_size,filter_sizes,config)
        
    
    def cnn(self,sequence_length, num_classes, vocab_size,filter_sizes,config):
        l2_loss = tf.constant(0.0)
        self.config = config 
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.Variable = tf.Variable(
                tf.random_uniform([vocab_size, self.config.embedding_size], -1.0, 1.0),
                name="Variable")
            self.embedded_chars = tf.nn.embedding_lookup(self.Variable, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                Variable = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Variable")
                print(Variable)
                b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    Variable,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config.num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)
            
        with tf.name_scope("output"):
            Variable = tf.get_variable(
                "Variable",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(Variable)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, Variable, b, name="scores")
            self.y_pred_cls = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss
        
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name="accuracy")
