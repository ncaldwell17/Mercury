import sys
import math
import re
import random
import os
import pprint
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter

# helper functions
# reads the file as a list of words
def read_file(path_to_file):
    with open(path_to_file, "r") as text:
        ls_words = text.read().replace("\n", "<eos>").split()
    # Clean the vocab from random characters within the corpora
    regex = re.compile(r'[.a-zA-Z0-9]')
    if a3 == 'wiki':
        return [i.lower() for i in ls_words if (regex.search(i) or i == '<eos>')]
    else:
        return [i for i in ls_words if (regex.search(i) or i == '<eos>')]


# makes batches of data for later use
def make_batches(data, batch_size, window_size):
    x_data = []
    y_data = []
    for i in range(len(data)):
        if i > window_size - 1:
            x_data.append(data[i - window_size:i])
            y_data.append(data[i])
    batches = int(len(x_data) / batch_size)
    batch_out = list()
    for i in range(batches):
        # For each batch
        start_i = batch_size * i
        end_i = start_i + batch_size
        x_values = x_data[start_i:end_i]
        y_values = y_data[start_i:end_i]
        batch_out.append([x_values, y_values])
    return batch_out


# brown corpus is huge, splits according to Bengio layout in paper
def split_brown():
    with open('data/brown.txt') as file:
        text_list = file.read().split()
    training = ' '.join(text_list[:800000])
    training_file = open("data/brown.train.txt", "w")
    training_file.write(training)
    training_file.close()

    validation = ' '.join(text_list[800000:1000000])
    validation_file = open("data/brown.valid.txt", "w")
    validation_file.write(validation)
    validation_file.close()

    testing = ' '.join(text_list[1000000:])
    testing_file = open("data/brown.test.txt", "w")
    testing_file.write(testing)
    testing_file.close()

class Preprocessor:
    
    def __init__(self, path):
        raw_list = read_file(path)
        top_words = Counter(raw_list).most_common()
        words = [word[0] for word in top_words if word[1] >= 3]
        if '<unk>' in words:
            words.remove('<unk>')
        self.word_dict = {'<unk>': 0}
        for i in range(1, len(words)):
            self.word_dict[words[i]] = i
        self.vocab_size = len(self.word_dict)
        self.word_dict_reverse = dict(zip(self.word_dict.values(), self.word_dict.keys()))
        self.text_as_index = []
        for word in words:
            idx = 0
            if word in self.word_dict:
                idx = self.word_dict[word]
            self.text_as_index.append(idx)

    def generate_data(self, path):
        words = read_file(path)
        text_as_index = []
        for word in words:
            idx = 0
            if word in self.word_dict:
                    idx = self.word_dict[word]
            text_as_index.append(idx)
        return text_as_index

class BengioModel:

    def __init__(self):
        self.batch_size = 256
        self.embedding_size = config['embedding_size']
        self.window_size = config['window_size']
        self.hidden_layers = config['hidden_layers']

    def train(self, train_data, validate_data, num_epochs=20):
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            device = '/gpu:0'
            print('Using GPU')
        else:
            device = '/cpu:0'
            print('Using CPU')
        with tf.device(device):
            # C/P everything from def train_model above until SGD optimizer
            # all code that I want to run on GPU insert here
            self.x_input = tf.placeholder(tf.int64, [None, self.window_size])
            self.y_input = tf.placeholder(tf.int64, [None])
            
            z = self.embedding_size * self.window_size
            
            # hidden layer biases
            d = tf.Variable(tf.random_uniform([self.hidden_units]))
            # output biases
            b = tf.Variable(tf.random_uniform([vocab_size]))
            
            # weights
            # C matrix function
            word_embeddings = tf.Variable(tf.random_uniform([vocab_size,
                                                             self.embedding_size],
                                                            -1.0,
                                                            1.0))
            flattened_exes = tf.layers.flatten(self.x_input)
            lookup = tf.nn.embedding_lookup(word_embeddings, flattened_exes)
            xt = tf.reshape(lookup, [self.batch_size, z])
                                                            
            # H Weight
            H = tf.Variable(tf.truncated_normal([z, self.hidden_units],
                                                stddev=1.0 / math.sqrt(z)))
            # W Weight
            W = tf.Variable(tf.truncated_normal([z, vocab_size],
                                                stddev=1.0 / math.sqrt(z)))
            # U Weight
            U = tf.Variable(tf.truncated_normal([self.hidden_units, vocab_size],
                                                stddev=1.0 / vocab_size))
                                                            
            # embed is [n*b, embedding_size]
            hidden_out = tf.nn.bias_add(tf.matmul(xt, H), d)
            tan_out = tf.nn.tanh(hidden_out)
            y_logits = tf.nn.bias_add(tf.matmul(xt, W), b) + tf.matmul(tan_out, U)
            
            y_pred = tf.nn.softmax(y_logits)
            y_pred_cls = tf.argmax(y_pred, axis=1)

            y_one_hot = tf.one_hot(self.y_input, vocab_size)
            self.ce_result = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_logits, labels=y_one_hot))

            # Construct the SGD optimizer.
            learn_rate = 0.002
            beta1 = 0.9
            beta2 = 0.999
            optimizer = tf.train.AdamOptimizer(learn_rate, beta1, beta2).minimize(self.ce_result)
            correct_prediction = tf.equal(y_pred_cls, self.y_input)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            config = tf.ConfigProto(allow_soft_placement=True)
            self.session = tf.Session(config=config)
            self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            print('Training....')
            global acc_hist_train, cost_hist_train
            patience = 2
            for i in range(num_epochs):
                batches = generate_batches(train_data, self.batch_size, self.window_size)
                total_batches = len(batches)
                batch_count = 0
                last_complete = 0
                num_messages = 10  # number of printouts per epoch
                for batch in batches:
                    batch_count += 1
                    x_batch = batch[0]
                    y_input_batch = batch[1]
                    feed_dict_train = {self.x_input: x_batch,
                                       self.y_input: y_input_batch}
                    self.session.run(optimizer, feed_dict=feed_dict_train)
                    completion = 100 * batch_count / total_batches
                    if batch_count % (int(total_batches / num_messages)) == 0:
                        print('Epoch #%2d-   Batch #%5d:   %4.2f %% completed.' % (i + 1, batch_count, completion))
                        a_t, c_t = self.test(train_data)
                        a, c = self.test(validate_data)
                        acc_hist_train.append(a)
                        cost_hist_train.append(c)

        print('Training Complete')
        save_path = saver.save(self.session, "../models/" + a2 + '_' + a3 + ".ckpt")
        print("Model saved in path: %s" % save_path)
        return

    def restore(self, model_path):
        if tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
            device = '/gpu:0'
            print('Using GPU')
        else:
            device = '/cpu:0'
            print('Using CPU')
        with tf.device(device):
            # C/P everything from def train_model above until SGD optimizer
            # all code that I want to run on GPU insert here
            self.x_input = tf.placeholder(tf.int64, [None, self.window_size])
            self.y_input = tf.placeholder(tf.int64, [None])
            
            z = self.embedding_size * self.window_size
            
            # hidden layer biases
            d = tf.Variable(tf.random_uniform([self.hidden_layers]))
            # output biases
            b = tf.Variable(tf.random_uniform([vocab_size]))
            
            # weights
            # C matrix function
            word_embeddings = tf.Variable(tf.random_uniform([vocab_size,
                                                             self.embedding_size],
                                                            -1.0,
                                                            1.0))
            flattened_exes = tf.layers.flatten(self.x_input)
            lookup = tf.nn.embedding_lookup(word_embeddings, flattened_exes)
            xt = tf.reshape(lookup, [self.batch_size, z])
            
            # H Weight
            H = tf.Variable(tf.truncated_normal([z, self.hidden_layers],
                                                stddev=1.0 / math.sqrt(z)))
            # W Weight
            W = tf.Variable(tf.truncated_normal([z, vocab_size],
                                                stddev=1.0 / math.sqrt(z)))
            # U Weight
            U = tf.Variable(tf.truncated_normal([self.hidden_layers, vocab_size],
                                                stddev=1.0 / vocab_size))
            
            # embed is [n*b, embedding_size]
            hidden_out = tf.nn.bias_add(tf.matmul(xt, H), d)
            tan_out = tf.nn.tanh(hidden_out)
            y_logits = tf.nn.bias_add(tf.matmul(xt, W), b) + tf.matmul(tan_out, U)
            
            y_pred = tf.nn.softmax(y_logits)
            y_pred_cls = tf.argmax(y_pred, axis=1)
            
            y_one_hot = tf.one_hot(self.y_input, vocab_size)
            self.ce_result = tf.reduce_mean(
                                                tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_logits, labels=y_one_hot))

            # Construct the SGD optimizer.
            learn_rate = 0.002
            beta1 = 0.9
            beta2 = 0.999
            optimizer = tf.train.AdamOptimizer(learn_rate, beta1, beta2).minimize(self.ce_result)
            correct_prediction = tf.equal(y_pred_cls, self.y_input)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver()

            with tf.Session() as sess:
                # Restore variables from disk.
                saver.restore(sess, model_path)
                print("Model restored.")
                test_batches = generate_batches(test_data, self.batch_size, self.window_size)
                cost, acc = [], []
                for batch in test_batches:
                    feed_dict_test = {self.x_input: batch[0],
                                      self.y_input: batch[1]}
                    acc.append(sess.run(self.accuracy, feed_dict=feed_dict_test))
                    cost.append(sess.run(self.cross_entropy, feed_dict=feed_dict_test))
                avg_acc = sum(acc) / float(len(acc))
                avg_cost = sum(cost) / float(len(cost))
                print("   Accuracy on test-set:   %4.2f %% \n" % (avg_acc * 100),
                      "   Cost on test-set:       %4.2f \n" % avg_cost,
                      "   Perplexity on test-set:       %4.2f \n" % np.exp(avg_cost))

    def test(self, test_data):
        test_batches = generate_batches(test_data, self.batch_size, self.window_size)
        cost, acc = [], []
        for batch in test_batches:
            feed_dict_test = {self.x_input: batch[0],
                              self.y_input: batch[1]}
            acc.append(self.session.run(self.accuracy, feed_dict=feed_dict_test))
            cost.append(self.session.run(self.cross_entropy, feed_dict=feed_dict_test))
        avg_acc = sum(acc) / float(len(acc))
        avg_cost = sum(cost) / float(len(cost))
        print("   Accuracy on valid-set:   %4.2f %%" % (avg_acc * 100),
              "   Cost on valid-set:       %4.2f \n" % avg_cost)
        return avg_acc, avg_cost


a1 = sys.argv[1]
a2 = sys.argv[2]
a3 = sys.argv[3]

configs = {'MLP1': {'window_size': 5, 'hidden_layers': 50, 'embedding_size': 60, 'direct': True, 'mix': False},
           'MLP3': {'window_size': 5, 'hidden_layers': 0, 'embedding_size': 60, 'direct': True, 'mix': False},
           'MLP5': {'window_size': 5, 'hidden_layers': 0, 'embedding_size': 30, 'direct': True, 'mix': False},
           'MLP7': {'window_size': 3, 'hidden_layers': 50, 'embedding_size': 30, 'direct': True, 'mix': False},
           'MLP9': {'window_size': 5, 'hidden_layers': 100, 'embedding_size': 30, 'direct': False, 'mix': False},
           }
corpora = ['wiki', 'brown']

if __name__ == "__main__":
    if a1 not in ['train', 'load'] or a2 not in configs or a3 not in corpora:
        print('Request not recognized')
        sys.exit()
    elif a1 == 'train':
        if a3 == 'wiki':
            path_train = "data/wiki.train.txt"
            path_validate = "data/wiki.valid.txt"
            path_test = "data/wiki.test.txt"
        elif a3 == 'brown':
            split_brown()
            path_train = "data/brown.train.txt"
            path_validate = "data/brown.valid.txt"
            path_test = "data/brown.test.txt"


        config = configs[a2]
        corpus = Preprocessor(path_train)
        vocab_size = corpus.vocab_size
        train_data = corpus.generate_data(path_train)
        validate_data = corpus.generate_data(path_validate)
        test_data = corpus.generate_data(path_test)
        model = BengioModel()
        acc_hist_train, cost_hist_train = [.1] * 10, [7] * 10
        model.train(train_data, validate_data)
        plot_learning(acc_hist_train[10:], cost_hist_train[10:])

    elif a1 == 'load':
        if a3 == 'wiki':
            path_train = "data/wiki.train.txt"
            path_validate = "data/wiki.valid.txt"
            path_test = "data/wiki.test.txt"
        elif a3 == 'brown':
            split_brown()
            path_train = "data/brown.train.txt"
            path_validate = "data/brown.valid.txt"
            path_test = "data/brown.test.txt"

        config = configs[a2]
        corpus = Preprocessor(path_train)
        vocab_size = corpus.vocab_size
        test_data = corpus.generate_data(path_test)
        model = BengioModel()
        model.restore('models/' + a2 + '_' + a3 + '.ckpt')
