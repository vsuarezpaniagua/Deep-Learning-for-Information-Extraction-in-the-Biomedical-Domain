#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import time
import datetime
import data_load
from data_load import print_results
from tensorflow.contrib import learn
from nltk import word_tokenize
#word_tokenize = list # char-level
def tokenizer_fn(iterator):
    return [(yield word_tokenize(value)) for value in iterator]
def tokenizer_pos_fn(iterator):
    return [(yield value.split(" ")) for value in iterator]
from deepLearningModel import DL_RE
from nltk.data import load
from gensim.models import KeyedVectors

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("train_path", "", "Data source for the train data")
tf.flags.DEFINE_string("dev_path", "", "Data source for the dev data")
tf.flags.DEFINE_float("dev_percentage", 0.0, "Percentage [0-1] of the training data to use for validation (default: 0.0)")
tf.flags.DEFINE_boolean("generation", False, "Generate synthetic data (default: False)")
tf.flags.DEFINE_boolean("filtered", False, "Use negative filtering (default: False)")
tf.flags.DEFINE_boolean("sampling", False, "Use random sampling to balance the dataset (default: False)")
tf.flags.DEFINE_boolean("weighted", False, "Use weighted cross entropy (default: False)")

# Model Hyperparameters
tf.flags.DEFINE_string("embedding_path", "", "Load pretrained embedding model (default: '')")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("pos_dim", 0, "Dimensionality of part-of-speech embedding (default: 0)")
tf.flags.DEFINE_integer("position_dim", 0, "Dimensionality of position embedding (default: 0)")
tf.flags.DEFINE_integer("type_dim", 0, "Dimensionality of type embedding (default: 0)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '2,3,4')")
tf.flags.DEFINE_string("num_filters", "200", "Comma-separated number of filters per filter size (default: 200)")
tf.flags.DEFINE_string("neurons", "0", "Comma-separated number of neurons per fully connected layer (default: 0)")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Adam Optimizer learning rate (default: 0.001)")
tf.flags.DEFINE_float("max_grad_norm", 0.0, "Norm gradient clipping (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 25)")
tf.flags.DEFINE_integer("patience", 10, "Number of patience epochs for early stopping criteria (default: 10)")
tf.flags.DEFINE_string("checkpoint_file", datetime.datetime.now().isoformat().split(".")[0].replace(":","_"), "Checkpoint file for saving the training model (default: 'datetime.now')")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:"+"\n".join([(attr.upper()+"="+str(FLAGS[attr].value)) for attr in sorted(FLAGS)])+"\n")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text_train, y_train_ids, distances_train, pos_tags_train, sdp_train = data_load.load_data(FLAGS.train_path)

# Save data
print("...Saving data")
with open("data_labels_train.txt", 'w+') as f:
    f.write("\n".join([x_text_train[i] + '\t' + y_train_ids[i,0] for i in range(len(x_text_train))]))

# One-hot labels
y_train = y_train_ids[:,0]
labelsnames = np.unique(y_train).tolist()
y_train = np.eye(len(labelsnames), dtype=int)[np.searchsorted(labelsnames, y_train)]#np.eye(len(labelsnames), dtype=int)[[labelnames.index(i) for i in y_train]]
# Build the word vocabulary
max_document_length = max([len(word_tokenize(x)) for x in x_text_train])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=tokenizer_fn)#min_frequency = 100
x_train = np.array(list(vocab_processor.fit_transform(x_text_train)))
# Build the POS vocabulary
vocab_processor_pos = learn.preprocessing.VocabularyProcessor(max_document_length, tokenizer_fn=tokenizer_pos_fn)
vocab_processor_pos.fit(list(load('help/tagsets/upenn_tagset.pickle').keys()))
pos_train = np.array(list(vocab_processor_pos.transform(pos_tags_train)))
# Distance1 and Distance2
d1_train = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - np.array(distances_train)[:,0][:,None]
d2_train = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - np.array(distances_train)[:,1][:,None]
# Build the Type vocabulary
labelstypes = np.unique(y_train_ids[:,[3,4]]).tolist()
type_train = np.zeros(x_train.shape, dtype=np.int64)
type_train[[range(len(type_train)), np.array(distances_train)[:,0]]] = np.searchsorted(labelstypes, y_train_ids[:,3])+1
type_train[[range(len(type_train)), np.array(distances_train)[:,1]]] = np.searchsorted(labelstypes, y_train_ids[:,4])+1

# Remove filtered instances
if FLAGS.filtered:
    neg_indices = [data_load.filter_neg(x_text_train[i], y_train_ids[i,1], y_train_ids[i,2]) for i in range(len(x_text_train))]
    #neg_data = np.array(list(zip(x_text_train, y_train_ids, y_train, pos_train, d1_train, d2_train, type_train)))[neg_indices]
    #x_text_train, y_train_ids, y_train, pos_train, d1_train, d2_train, type_train = zip(*neg_data)
    x_text_train_filtered = x_text_train[neg_indices]
    y_train_ids_filtered = y_train_ids[neg_indices]
    x_train_filtered = x_train[neg_indices]
    y_train_filtered = y_train[neg_indices]
    pos_train_filtered = pos_train[neg_indices]
    d1_train_filtered = d1_train[neg_indices]
    d2_train_filtered = d2_train[neg_indices]
    type_train_filtered = type_train[neg_indices]
    pos_indices = np.logical_not(neg_indices)
    #pos_data = np.array(list(zip(x_text_train, y_train_ids, y_train, pos_train, d1_train, d2_train, type_train)))[pos_indices]
    #x_text_train, y_train_ids, y_train, pos_train, d1_train, d2_train, type_train = zip(*pos_data)
    x_text_train = x_text_train[pos_indices]
    y_train_ids = y_train_ids[pos_indices]
    x_train = x_train[pos_indices]
    y_train = y_train[pos_indices]
    pos_train = pos_train[pos_indices]
    d1_train = d1_train[pos_indices]
    d2_train = d2_train[pos_indices]
    type_train = type_train[pos_indices]
    
'''
# Delete repeated examples
x_train, unique_indices = np.unique(x_train, return_index=True, axis=0)
#unique_data = np.array(list(zip(x_text_train, y_train_ids, y_train, pos_train, d1_train, d2_train, type_train)))[unique_indices]
#x_text_train, y_train_ids, y_train, pos_train, d1_train, d2_train, type_train = zip(*unique_data)
x_text_train = x_text_train[unique_indices]
y_train_ids = y_train_ids[unique_indices]
y_train = y_train[unique_indices]
pos_train = pos_train[unique_indices]
d1_train = d1_train[unique_indices]
d2_train = d2_train[unique_indices]
type_train = type_train[unique_indices]
'''

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(range(len(y_train)))
#shuffled_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))[shuffle_indices]
#x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train = zip(*shuffled_data)
x_text_train = x_text_train[shuffle_indices]
y_train_ids = y_train_ids[shuffle_indices]
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]
pos_train = pos_train[shuffle_indices]
d1_train = d1_train[shuffle_indices]
d2_train = d2_train[shuffle_indices]
type_train = type_train[shuffle_indices]

'''
# Remove padding
seq_train = np.array([len(word_tokenize(i)) for i in x_text_train])
x_train = np.array([x_train[i][:seq_train[i]] for i in range(len(x_train))])
pos_train = np.array([pos_train[i][:seq_train[i]] for i in range(len(pos_train))])
d1_train = np.array([d1_train[i][:seq_train[i]] for i in range(len(d1_train))])
d2_train = np.array([d2_train[i][:seq_train[i]] for i in range(len(d2_train))])
type_train = np.array([type_train[i][:seq_train[i]] for i in range(len(type_train))])
'''

# Split in Train-Dev sets
# ==================================================

if FLAGS.dev_path:
    # Load data
    x_text_dev, y_dev_ids, distances_dev, pos_tags_dev, sdp_dev = data_load.load_data(FLAGS.dev_path)
    # Save data
    with open("data_labels_dev.txt", 'w+') as f:
        f.write("\n".join([x_text_dev[i] + '\t' + y_dev_ids[i,0] for i in range(len(x_text_dev))]))
    # One-hot labels
    y_dev = y_dev_ids[:,0]
    y_dev = np.eye(len(labelsnames), dtype=int)[np.searchsorted(labelsnames, y_dev)]
    # Map data into word vocabulary
    x_dev = np.array(list(vocab_processor.transform(x_text_dev)))
    # Map data into POS vocabulary
    pos_dev = np.array(list(vocab_processor_pos.transform(pos_tags_dev)))
    # Distance1 and Distance2
    d1_dev = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - distances_dev[:,0][:,None]
    d2_dev = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - distances_dev[:,1][:,None]
    # Build the Type vocabulary
    type_dev = np.zeros(x_dev.shape, dtype=np.int64)
    type_dev[[range(len(type_dev)), np.array(distances_dev)[:,0]]] = np.searchsorted(labelstypes, y_dev_ids[:,2])+1
    type_dev[[range(len(type_dev)), np.array(distances_dev)[:,1]]] = np.searchsorted(labelstypes, y_dev_ids[:,4])+1
    # Remove filtered instances
    if FLAGS.filtered:
        neg_indices = [data_load.filter_neg(x_text_dev[i], y_dev_ids[i,1], y_dev_ids[i,3]) for i in range(len(x_text_dev))]
        #neg_data = np.array(list(zip(x_text_dev, y_dev_ids, y_dev, pos_dev, d1_dev, d2_dev, type_dev)))[neg_indices]
        #x_text_dev, y_dev_ids, x_dev, y_dev, pos_dev, d1_dev, d2_dev, type_dev = zip(*neg_data)
        x_text_dev_filtered = x_text_dev[neg_indices]
        y_dev_ids_filtered = y_dev_ids[neg_indices]
        x_dev_filtered = x_dev[neg_indices]
        y_dev_filtered = y_dev[neg_indices]
        pos_dev_filtered = pos_dev[neg_indices]
        d1_dev_filtered = d1_dev[neg_indices]
        d2_dev_filtered = d2_dev[neg_indices]
        type_dev_filtered = type_dev[neg_indices]
        pos_indices = np.logical_not(neg_indices)
        #pos_data = np.array(list(zip(x_text_dev, y_dev_ids, x_dev, y_dev, pos_dev, d1_dev, d2_dev, type_dev)))[pos_indices]
        #x_text_dev, y_dev_ids, x_dev, y_dev, pos_dev, d1_dev, d2_dev, type_dev = zip(*pos_data)
        x_text_dev = x_text_dev[pos_indices]
        y_dev_ids = y_dev_ids[pos_indices]
        x_dev = x_dev[pos_indices]
        y_dev = y_dev[pos_indices]
        pos_dev = pos_dev[pos_indices]
        d1_dev = d1_dev[pos_indices]
        d2_dev = d2_dev[pos_indices]
        d2_dev = d2_dev[pos_indices]
elif FLAGS.dev_percentage:
    '''
    # Split train/dev set
    dev_sample_index = -1 * int(FLAGS.dev_percentage * float(len(y_train)))
    x_text_train, x_text_dev = x_text_train[:dev_sample_index], x_text_train[dev_sample_index:]
    x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]
    pos_train, pos_dev = pos_train[:dev_sample_index], pos_train[dev_sample_index:]
    d1_train, d1_dev = d1_train[:dev_sample_index], d1_train[dev_sample_index:]
    d2_train, d2_dev = d2_train[:dev_sample_index], d2_train[dev_sample_index:]
    type_train, type_dev = type_train[:dev_sample_index], type_train[dev_sample_index:]
    '''
    # Split train/dev set with the same probability for each class
    x_text_dev = np.array([], dtype=np.int64)
    y_dev_ids = np.array([], dtype=np.int64).reshape(0, y_train_ids.shape[1])
    x_dev = np.array([], dtype=np.int64).reshape(0, x_train.shape[1])
    y_dev = np.array([], dtype=np.int64).reshape(0, y_train.shape[1])
    pos_dev = np.array([], dtype=np.int64).reshape(0, pos_train.shape[1])
    d1_dev = np.array([], dtype=np.int64).reshape(0, d1_train.shape[1])
    d2_dev = np.array([], dtype=np.int64).reshape(0, d2_train.shape[1])
    type_dev = np.array([], dtype=np.int64).reshape(0, type_train.shape[1])
    for c in range(len(labelsnames)):
        indices = np.where(np.argmax(y_train,1)==c)[0]
        # Add the indices to dev set
        dev_indices = indices[-1 * int(FLAGS.dev_percentage * float(len(indices))):]
        #TODO: no vale como se inicializan los vectores dev. no vale x_text_dev porque es sequence
        #dev_data = np.array(list(zip(x_text_dev, y_dev_ids, x_dev, y_dev, pos_dev, d1_dev, d2_dev, type_dev)))
        #train_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))[dev_indices]
        #x_text_dev, y_dev_ids, x_dev, y_dev, pos_dev, d1_dev, d2_dev, d2_dev = zip(*np.concatenate([dev_data, train_data]))
        x_text_dev = np.concatenate([x_text_dev, x_text_train[dev_indices]])
        y_dev_ids = np.concatenate([y_dev_ids, y_train_ids[dev_indices]])
        x_dev = np.concatenate([x_dev, x_train[dev_indices]])
        y_dev = np.concatenate([y_dev, y_train[dev_indices]])
        pos_dev = np.concatenate([pos_dev, pos_train[dev_indices]])
        d1_dev = np.concatenate([d1_dev, d1_train[dev_indices]])
        d2_dev = np.concatenate([d2_dev, d2_train[dev_indices]])
        type_dev = np.concatenate([type_dev, type_train[dev_indices]])
        # Remove the indices from the train set 
        train_indices = np.array([True]*len(y_train))
        train_indices[dev_indices] = False
        #train_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))[train_indices]
        #x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train = zip(*train_data)
        x_text_train = x_text_train[train_indices]
        y_train_ids = y_train_ids[train_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        pos_train = pos_train[train_indices]
        d1_train = d1_train[train_indices]
        d2_train = d2_train[train_indices]
        type_train = type_train[train_indices]
        
# Generation of sentences
# ==================================================

if FLAGS.generation:
    if not os.path.exists("generation"):
        os.makedirs("generation")
    for c in range(len(labelsnames)):
        indices = np.where(np.argmax(y_train,1)==c)[0]
        indices_dev = np.where(np.argmax(y_dev,1)==c)[0]
        if not labelsnames[c] == 'None':
            x_text_gen, y_gen_ids, distances_gen, pos_tags_gen, sdp_gen = data_load.text_generation(x_text_train[indices], labelsnames[c], x_text_dev[indices_dev])
            # One-hot labels
            y_gen = y_gen_ids[:,0]
            y_gen = np.eye(len(labelsnames), dtype=int)[np.searchsorted(labelsnames, y_gen)]
            # Map data into word vocabulary
            x_gen = np.array(list(vocab_processor.transform(x_text_gen)))
            # Map data into POS vocabulary
            pos_gen = np.array(list(vocab_processor_pos.transform(pos_tags_gen)))
            # Distance1 and Distance2
            d1_gen = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - distances_gen[:,0][:,None]
            d2_gen = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - distances_gen[:,1][:,None]
            # Build the Type vocabulary
            type_gen = np.zeros(x_gen.shape, dtype=np.int64)
            # type_gen[[range(len(type_gen)), np.array(distances_gen)[:,0]]] = np.searchsorted(labelstypes, y_gen_ids[:,2])+1
            # type_gen[[range(len(type_gen)), np.array(distances_gen)[:,1]]] = np.searchsorted(labelstypes, y_gen_ids[:,4])+1
            #train_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))
            #gen_data = np.array(list(zip(x_text_gen, y_gen_ids, x_gen, y_gen, pos_gen, d1_gen, d2_gen, type_gen)))
            #x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train = zip(*np.concatenate([train_data, gen_data]))
            x_text_train = np.concatenate([x_text_train, x_text_gen])
            y_train_ids = np.concatenate([y_train_ids, y_gen_ids])
            x_train = np.concatenate([x_train, x_gen])
            y_train = np.concatenate([y_train, y_gen])
            pos_train = np.concatenate([pos_train, pos_gen])
            d1_train = np.concatenate([d1_train, d1_gen])
            d2_train = np.concatenate([d2_train, d2_gen])
            type_train = np.concatenate([type_train, type_gen])
            
# Balance the dataset
# ==================================================

if FLAGS.sampling:
    classes_proportion = np.max(np.sum(y_train,0)[np.logical_not(np.array(labelsnames)=='None')])/(np.sum(y_train,0)*1.0)-1
    for c in range(len(classes_proportion)):
        indices = np.where(np.argmax(y_train,1)==c)[0]
        if classes_proportion[c] > 0:
            # Add examples with Random Over Sampling
            keep_indices = indices*(int(classes_proportion[c]))+indices[:int(len(indices)*(classes_proportion[c]-int(classes_proportion[c])))]
            #dev_data = np.array(list(zip(x_text_dev, y_dev_ids, x_dev, y_dev, pos_dev, d1_dev, d2_dev, type_dev)))
            #train_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))[dev_indices]
            #x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train = zip(*np.concatenate([dev_data, train_data]))
            x_text_train = np.concatenate([x_text_train, x_text_train[keep_indices]])
            y_train_ids = np.concatenate([y_train_ids, y_train_ids[keep_indices]])
            x_train = np.concatenate([x_train, x_train[keep_indices]])
            y_train = np.concatenate([y_train, y_train[keep_indices]])
            pos_train = np.concatenate([pos_train, pos_train[keep_indices]])
            d1_train = np.concatenate([d1_train, d1_train[keep_indices]])
            d2_train = np.concatenate([d2_train, d2_train[keep_indices]])
            type_train = np.concatenate([type_train, type_train[keep_indices]])
        elif np.array(labelsnames)[c] == 'None' and (len(classes_proportion)-1) * (classes_proportion[c]+1) < 1:
            # Remove examples with Random Under Sampling
            removed_indices = indices[:-1 * int(((len(classes_proportion)-1) * (classes_proportion[c]+1)-1) * float(len(indices)))]
            keep_indices = np.array([True]*len(y_train))
            keep_indices[removed_indices] = False
            #keep_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))[keep_indices]
            #x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train = zip(*keep_data)
            x_text_train = x_text_train[keep_indices]
            y_train_ids = y_train_ids[keep_indices]
            x_train = x_train[keep_indices]
            y_train = y_train[keep_indices]
            pos_train = pos_train[keep_indices]
            d1_train = d1_train[keep_indices]
            d2_train = d2_train[keep_indices]
            type_train = type_train[keep_indices]
            
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(range(len(y_train)))
#shuffled_data = np.array(list(zip(x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train)))[shuffle_indices]
#x_text_train, y_train_ids, x_train, y_train, pos_train, d1_train, d2_train, type_train = zip(*shuffled_data)
x_text_train = x_text_train[shuffle_indices]
y_train_ids = y_train_ids[shuffle_indices]
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]
pos_train = pos_train[shuffle_indices]
d1_train = d1_train[shuffle_indices]
d2_train = d2_train[shuffle_indices]
type_train = type_train[shuffle_indices]

print("Total number of train examples: {}".format(len(y_train)))
if FLAGS.dev_percentage or FLAGS.dev_path:
    print("Total number of dev examples: {}".format(len(y_dev)))
    
# Weight of the classes in the dataset
# ==================================================

w_classes = [1.0]*y_train.shape[1]
if FLAGS.weighted:
    #w_classes = np.max(np.sum(y_train,0))/(np.sum(y_train,0)*1.0)
    #w_classes = np.min(np.sum(y_train,0))/(np.sum(y_train,0)*1.0)
    w_classes = np.sum(np.sum(y_train,0))/(np.sum(y_train,0)*1.0)
    
# Embedding Matrix
# ==================================================

# Create the embedding matrix
if FLAGS.embedding_dim:
    # Random embedding weights
    embedding_size = FLAGS.embedding_dim
    embedding_vocab = len(vocab_processor.vocabulary_)
    embedding_matrix = np.random.uniform(-1.0, +1.0, [embedding_vocab, embedding_size]).astype('float32')
if FLAGS.embedding_path:
    # Load Word2Vec embedding
    embedding_model = KeyedVectors.load_word2vec_format(FLAGS.embedding_path, binary=False)
    embedding_model.vocab = dict((k.lower(), v) for k,v in embedding_model.vocab.items())
    # Word2Vec embedding weights
    embedding_size = embedding_model.vector_size
    embedding_vocab = len(vocab_processor.vocabulary_)
    embedding_matrix = np.random.uniform(-1.0, +1.0, [embedding_vocab, embedding_size]).astype('float32')#1.0*np.random.randn(embedding_vocab, embedding_size).astype('float32')
    for word in vocab_processor.vocabulary_._reverse_mapping:
        if word in embedding_model.vocab:
            idx = vocab_processor.vocabulary_._mapping[word]
            embedding_matrix[idx] = embedding_model[word]
            
# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = DL_RE(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            embedding_matrix = embedding_matrix,
            POS_vocab = len(vocab_processor_pos.vocabulary_),
            POS_size = FLAGS.pos_dim,
            position_vocab = 2*max_document_length-1,
            position_size = FLAGS.position_dim,
            type_vocab = len(labelstypes)+1,
            type_size = FLAGS.type_dim,
            num_filters = list(map(int, FLAGS.num_filters.split(","))),
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            neurons = list(map(int, FLAGS.neurons.split(","))),
            l2_reg_lambda = FLAGS.l2_reg_lambda,
            class_weights = w_classes)
        
        # Define Training procedure
        #learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads, tvars = zip(*optimizer.compute_gradients(model.loss))
        if FLAGS.max_grad_norm:
            grads, _ = tf.clip_by_global_norm(grads, FLAGS.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
        '''
        grads_and_vars = optimizer.compute_gradients(model.loss)
        if FLAGS.max_grad_norm:
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v) for g, v in grads_and_vars]
        train_op = optimizer.apply_gradient(grads_and_vars, global_step=model.global_step)
        '''
        
        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in zip(grads, tvars):#grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        
        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.checkpoint_file))
        print("Writing to {}\n".format(out_dir))
        
        # Checkpoint directory
        checkpoint_dir = os.path.join(out_dir, "checkpoints")
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_epochs)
        
        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        vocab_processor_pos.save(os.path.join(out_dir, "vocab_pos"))
        # Write labels names
        np.save(os.path.join(out_dir, "labels.npy"), np.array(labelsnames))
        # Write labels types
        np.save(os.path.join(out_dir, "labelstypes.npy"), np.array(labelstypes))
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)
        
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        # Dev Summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        
        # Embedding Summaries
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embed_summary_writer = tf.summary.FileWriter(checkpoint_prefix, sess.graph)
        Wembedding = config.embeddings.add()
        Wembedding.tensor_name = model.Wembedding.name
        Wembedding.metadata_path = os.path.join(checkpoint_prefix, 'Wembedding.tsv')
        with open(Wembedding.metadata_path, 'w') as f:
            f.write("\n".join(vocab_processor.vocabulary_._reverse_mapping))
        if FLAGS.pos_dim:
            Wpos = config.embeddings.add()
            Wpos.tensor_name = model.Wpos.name
            Wpos.metadata_path = os.path.join(checkpoint_prefix, 'Wpos.tsv')
            with open(Wpos.metadata_path, 'w') as f:
                f.write("\n".join(vocab_processor_pos.vocabulary_._reverse_mapping))
        if FLAGS.position_dim:
            Wd1 = config.embeddings.add()
            Wd1.tensor_name = model.Wd1.name
            Wd1.metadata_path = os.path.join(checkpoint_prefix, 'Wd1.tsv')
            with open(Wd1.metadata_path, 'w') as f:
                f.write("\n".join([str(i) for i in np.arange(2*max_document_length-1)]))
            Wd2 = config.embeddings.add()
            Wd2.tensor_name = model.Wd2.name
            Wd2.metadata_path = os.path.join(checkpoint_prefix, 'Wd2.tsv')
            with open(Wd2.metadata_path, 'w') as f:
                f.write("\n".join([str(i) for i in np.arange(2*max_document_length-1)]))
        if FLAGS.type_dim:
            Wtype = config.embeddings.add()
            Wtype.tensor_name = model.Wtype.name
            Wtype.metadata_path = os.path.join(checkpoint_prefix, 'Wtype.tsv')
            with open(Wtype.metadata_path, 'w') as f:
                f.write("\n".join(['None']+labelstypes))
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(embed_summary_writer, config)
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        def model_step(x, y, pos, d1, d2, etype, dropout, training, summary=None, writer=None, verbose=False):
            feed_dict = {
              model.input_x: x,
              model.input_y: y,
              model.input_POS: pos,
              model.input_distance1: d1,
              model.input_distance2: d2,
              model.input_type: etype,
              model.dropout: dropout
            }
            if training:
                train_op.run(feed_dict)
            step, loss, accuracy, predictions = sess.run([model.global_step, model.loss, model.accuracy, model.predictions], feed_dict)
            f1 = print_results(np.argmax(y, 1), predictions, labelsnames, verbose=verbose)[0][3]
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, F1 {:g}".format(time_str, step, loss, accuracy, f1))
            if summary is not None:
                summaries = summary.eval(feed_dict)
                if writer is not None:
                    writer.add_summary(summaries, step)
            return step, f1
        
        # Training loop
        best, best_step, patience = 0, 0, 0
        for epoch in range(FLAGS.num_epochs):
            # Generate batches
            shuffle_indices = np.random.permutation(range(len(y_train)))
            shuffled_data = np.array(list(zip(x_train, y_train, pos_train, d1_train, d2_train, type_train)))[shuffle_indices]
            for batch_num in range(int((len(y_train) - 1) / FLAGS.batch_size) + 1):
                start_index, end_index = batch_num * FLAGS.batch_size, min((batch_num + 1) * FLAGS.batch_size, len(y_train))
                x_batch, y_batch, pos_batch, d1_batch, d2_batch, type_batch = zip(*shuffled_data[start_index:end_index])
                step, f1 = model_step(x_batch, y_batch, pos_batch, d1_batch, d2_batch, type_batch, FLAGS.dropout, True, summary=train_summary_op, writer=train_summary_writer)
            # Validation Step
            if FLAGS.dev_percentage or FLAGS.dev_path:
                print("\nEvaluation:")
                step, f1 = model_step(x_dev, y_dev, pos_dev, d1_dev, d2_dev, type_dev, 1.0, False, summary=dev_summary_op, writer=dev_summary_writer, verbose=True)
                if f1 > best:
                    best, best_step = f1, step
                    patience = 0
                else:
                    patience += 1
                if patience >= FLAGS.patience:
                    break
                # Save the model
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))
        print("Best step model = {} (F1 = {:g})\n".format(best_step, best))