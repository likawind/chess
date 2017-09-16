import tensorflow as tf;
import numpy as np;
from cnn_layers import *;
from read_data import *;
from global_names import *;
import hyper_parameters as hp;
import Tkinter;
import matplotlib.pyplot as plt;

#x 		=	tf.placeholder(tf.float32, shape=[None,8,8,7]);
#y_train =	tf.placeholder(tf.float32, shape=[None,1]);

board_batch, label_batch = read_dataset(filenames=[TFRECORD_FOLDER+str(i)+".tfrecord" for i in range(44)]);
is_training = tf.placeholder(tf.bool);

x = board_batch;
y_train = (tf.cast(tf.reshape(label_batch,[-1,1]), tf.float32)+1)/2;

#################################################################
x_1 = x;

b_1 = bias([32]);
W_1 = weight([3,3,7,32]);

#################################################################
x_2 = lrelu(batch_norm(conv2d(x_1, W_1)+b_1, is_training));

b_2 = bias([64]);
W_2 = weight([3,3,32,64]);

#################################################################
x_3 = lrelu(batch_norm(conv2d(x_2, W_2)+b_2, is_training));

b_3 = bias([128]);
W_3 = weight([3,3,64,128]);

#################################################################
x_4 = lrelu(batch_norm(conv2d(x_3, W_3)+b_3, is_training));

#concat feature vectors
dropout_keep_prob = tf.placeholder(tf.float32);
x_fc = tf.concat([tf.reshape(x_2,[-1,6*6*32]), tf.reshape(x_3,[-1,4*4*64]), tf.reshape(x_4,[-1,2*2*128])],1);
fc_feature_size = 6*6*32+4*4*64+2*2*128;

#fc_feature_size = 2*2*128;
#x_fc = tf.reshape(x_4,[-1,fc_feature_size]);

out_fc,reg = fc_dense(x_fc, fc_feature_size, [1024,512,512,1], dropout_keep_prob = dropout_keep_prob);
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_train, logits = out_fc));

n_paras = 3*3*7*32+3*3*32*64+3*3*64*128+fc_feature_size*1024+1024*512+512*512+512;

for w in reg:
	loss = loss + hp.Model["REG_RATE"]*tf.nn.l2_loss(w)/n_paras;

loss = loss + hp.Model["REG_RATE"]*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_2)+tf.nn.l2_loss(W_3))/n_paras;

#################################################################

if __name__ == "__main__":
	train_step = tf.train.AdamOptimizer(hp.Model["LEARNING_RATE"]).minimize(loss);

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer());
		coord = tf.train.Coordinator();
		threads = tf.train.start_queue_runners(coord=coord);
		losses = [];
		for i in range(hp.Model["NUM_RUN"]):
			_,loss_val = sess.run([train_step,loss],feed_dict={dropout_keep_prob:hp.Model["DROP_OUT_KEEP_PROB"], is_training:True});
			losses.append(loss_val);
			if i%100==0:
				print loss_val;
				print i;

		plt.plot(losses);
		plt.show();
		plt.show('hold');

		coord.request_stop();
		coord.join(threads);
