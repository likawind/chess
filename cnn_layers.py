import tensorflow as tf;
import hyper_parameters as hp;

def weight(shape):
	initial = tf.truncated_normal(shape,stddev=hp.Model["WEIGHT_INIT_STD"]);
	return tf.Variable(initial);

def bias(shape):
	initial = tf.truncated_normal(shape,stddev=hp.Model["WEIGHT_INIT_STD"]);
	return tf.Variable(initial);

def conv3d(x,W):
	return tf.nn.conv3d(x,W,strides=[1,1,1,1,1], padding='VALID');

def max_pool(x,size):
	return tf.nn.max_pool3d(x,ksize=[1,size,size,size,1],strides=[1,size,size,size,1],padding='SAME');

def fc_dense(x,input_size,hiddens,dropout_keep_prob=1):

	cur_input		= [];
	cur_input_size	= [];
	W 				= [];
	b 				= [];
	cur_input_dropped 	= [];
	cur_input.append(x);
	cur_input_size.append(input_size);
	for i in range(len(hiddens)):
		hidden_layer_size = hiddens[i];
		cur_input_dropped.append(tf.nn.dropout(cur_input[i], keep_prob = drop_out_keep_prob));
		W.append(weight([cur_input_size[i], hidden_layer_size]));
		b.append(bias([hidden_layer_size]));

		cur_input.append(tf.nn.relu(tf.matmul(cur_input_dropped[i], W[i])+b[i]));
		cur_input_size.append(hidden_layer_size);

	reg = W;

	return cur_input[len(hiddens)], reg;