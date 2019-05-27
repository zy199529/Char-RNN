#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zy19950209
# @Date:   2019-05-26 19:29:11
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-05-26 22:59:50
import tensorflow as tf


class CharRNN:

    def __init__(self, num_classes, num_seqs=64, num_steps=50, lstm_size=128, num_layers=2, learning_rate=0.001, grad_clip=5, sampling=False, train_keep_prob=0.5, use_embedding=False, embedding_size=128):
        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

    def bulid_inputs(self):
        with tf.name_scope('inputs'):
            # self.inputs-----num_seqs:一个batch内句子的个数=batch_size，num_steps：表示句子的长度
            self.inputs = tf.placeholder(
                tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(
                tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 中文需要embedding层
        # 英文字母没必要用embeddig层
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_calsses)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable(
                        'embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(
                        embedding, self.inputs)

# 定义LSTM模型：n:n

    def build_lstm(self):
            # 创建单个cell并堆叠

        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(
                lstm, output_keep_prob=keep_prob)
            return drop
        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)])
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)
        # 时间维度展开
            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(
                cell, self.lstm_inputs, initial_state=self.initial_state)
            seq_output = tf.concat(self.lstm_outputs, 1)  # 列连接
            # 把之前的list展开，成[batch, lstm_size*num_steps],然后 reshape, 成[batch*numsteps, lstm_size]
            # ??????不懂，将每个batch的每个state拼接成batch_size*lstm_size
            x = tf.reshape(seq_output, [-1, self.lstm_size])
            with tf.variable_scope('softmax'):
            	#softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
            	softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
            	softmax_b = tf.Variable(tf.zeros(self.num_classes))
            self.logits = tf.matmul(x, softmax_w)+softmax_b
            self.proba_presiction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)

    def bulid_optimizer(self):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))
