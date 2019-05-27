#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-05-26 18:45:11
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-05-26 19:29:17
import tensorflow as tf
import numpy as np
# rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# print(rnn_cell.state_size)

# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
# inputs = tf.placeholder(np.float32, shape=(32, 100))
# h0 = lstm_cell.zero_state(32, np.float32)
# output ,h1 = lstm_cell.call(inputs,h0)
# print(h1.h)
# print(h1.c)


# def get_a_cell():
#     return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# # 创建3层RNN
# cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
# print(cell.state_size)
# inputs = tf.placeholder(np.float32, shape=(32, 100))
# h0 = cell.zero_state(32, np.float32)
# output, h1 = cell.call(inputs, h0)
# print(h1)
