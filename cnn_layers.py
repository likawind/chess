import tensorflow as tf;
import hyper_parameters as hp;

def weight(shape):
	initial = tf.truncated_normal(shape,stddev=hp.Model["WEIGHT_INIT_STD"]);
	return tf.Variable(initial);

def bias(shape):
	initial = tf.truncated_normal(shape,stddev=hp.Model["WEIGHT_INIT_STD"]);
	return tf.Variable(initial);

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID');

def max_pool(x,size):
	return tf.nn.max_pool2d(x,ksize=[1,size,size,1],strides=[1,size,size,1],padding='SAME');

def lrelu(x):
	return tf.nn.relu(x)-hp.Model["RELU_LEAK_ALPHA"]*tf.nn.relu(-x);

def batch_norm(x, is_training = True):
	return tf.contrib.layers.batch_norm(x,
		is_training = is_training, 
		decay = hp.Model["BATCH_NORM"]["DECAY"], 
		epsilon = hp.Model["BATCH_NORM"]["EPS"], 
		renorm = hp.Model["BATCH_NORM"]["RENORM"],
		renorm_decay = hp.Model["BATCH_NORM"]["RENORM_DECAY"]);

def fc_dense(x,input_size,hiddens,dropout_keep_prob=1, is_training = True):

	cur_input		= [];
	cur_input_size	= [];
	W 				= [];
	b 				= [];
	cur_input_dropped 	= [];
	cur_input.append(x);
	cur_input_size.append(input_size);
	for i in range(len(hiddens)):
		hidden_layer_size = hiddens[i];
		cur_input_dropped.append(tf.nn.dropout(cur_input[i], keep_prob = dropout_keep_prob));
		W.append(weight([cur_input_size[i], hidden_layer_size]));
		b.append(bias([hidden_layer_size]));

		if i < len(hiddens)-1:
			cur_input.append(lrelu(batch_norm(tf.matmul(cur_input_dropped[i], W[i])+b[i], is_training = is_training)));
		else:
			#expose the last layer
			cur_input.append(tf.matmul(cur_input_dropped[i],W[i])+b[i]);
			
		cur_input_size.append(hidden_layer_size);

	reg = W;

	return cur_input[len(hiddens)], reg;