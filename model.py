import tensorflow as tf;
import numpy as np;
from cnn_layers import *;
from read_data import *;
import hyper_parameters as hp;

x 		=	tf.placeholder(tf.float32, shape=[None,8,8,7]);
y_train =	tf.placeholder(tf.float32, shape=[None,1]);

#################################################################
x_1 = x;

b_1 = bias([32]);
W_1 = weight([3,3,7,1,32]);

#################################################################
x_2 = tf.nn.relu(conv3d(x_1, W_1)+b_1);

b_2 = bias([64]);
W_2 = weight([3,3,1,32,64]);

#################################################################
x_3 = tf.nn.relu(conv3d(x_2, W_2)+b_2);

b_3 = bias([128]);
W_3 = weight([3,3,1,64,128]);

#################################################################
x_4 = tf.nn.relu(conv3d(x_3, W_3)+b_3);

#concat feature vectors
dropout_keep_prob = tf.placeholder(tf.float32);
x_fc = tf.concat(tf.reshape(x_2,[-1,6*6*32]), tf.reshape(x_3,[-1,4*4*64]), tf.reshape(x_4,[-1,2*2*128]));
fc_feature_size = 6*6*32+4*4*64+2*2*128;

out_fc,reg = fc_dense(x_fc, fc_feature_size, [1024,512,512,1], dropout_keep_prob = dropout_keep_prob);
loss = tf.nn.l2_loss(out_fc-y_train)+

