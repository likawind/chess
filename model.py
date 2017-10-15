import tensorflow as tf;
import numpy as np;
from cnn_layers import *;
from read_data import *;
from global_names import *;
import hyper_parameters as hp;
import Tkinter;
import matplotlib.pyplot as plt;
from tensorflow.contrib.layers import batch_norm as bn_layer;

class Agent:
	def __init__(self, train_data, train_label, 
		using_val = False, val_data = None, val_label = None):
		self.train_data = train_data;
		self.train_label = train_label;
		self.using_val = using_val;
		self.val_data = val_data;
		self.val_label = val_label;

		#initialize weight tensors
		conv_b = [];
		conv_W = [];
		fc_W = [];
		fc_b = [];

		self.fc_hiddens = [1024, 512, 512, 1];

		conv_b.append(bias([32]));
		conv_W.append(weight([3, 3, 7, 32])); 

		conv_b.append(bias([64]));
		conv_W.append(weight([3, 3, 32, 64]));

		conv_b.append(bias([128]));
		conv_W.append(weight([3, 3, 64, 128]));

		self.conv_layer_sizes = [8 * 8 * 7, 6 * 6 * 32, 4 * 4 * 64, 2 * 2 * 128];
		cur_input_size = 6 * 6 * 32+ 4 * 4 * 64 + 2 * 2 * 128;
		self.num_parameters = 3 *3 * 7 * 32 + 3 * 3 * 32 * 64 + 3 * 3 * 64 * 128;
		for hidden_layer_size in self.fc_hiddens:
			fc_W.append(weight([cur_input_size, hidden_layer_size]));
			self.num_parameters += cur_input_size * hidden_layer_size;
			fc_b.append(bias([hidden_layer_size]));
			cur_input_size = hidden_layer_size;

		self.parameters = {"conv_W": conv_W, "conv_b": conv_b, "fc_W": fc_W, "fc_b": fc_b};
		return;

	def network(self, input_batch, is_training, namescope = "DEFAULT_MODEL_NAMESCOPE"):
	#construct the network with the input tensor.	
		with tf.variable_scope(namescope, "model", reuse = None if is_training else True):
			cur_conv_input = [input_batch];
			conv_W = self.parameters["conv_W"];
			conv_b = self.parameters["conv_b"];
			fc_W = self.parameters["fc_W"];
			fc_b = self.parameters["fc_b"];

			dropout_keep_prob = hp.Model["DROP_OUT_KEEP_PROB"] if is_training else 1;
			for i in range(len(conv_b)):
				cur_conv_input.append(lrelu(
					bn_layer(
									conv2d(cur_conv_input[i], conv_W[i]) + conv_b[i], 
									is_training = is_training,
									decay = hp.Model["BATCH_NORM"]["DECAY"],
									epsilon = hp.Model["BATCH_NORM"]["EPS"]
					)
				)
			);

			conv_out = tf.concat(
				[tf.reshape(cur_conv_input[i], [-1,self.conv_layer_sizes[i]]) 
				 for i in range(1,len(cur_conv_input))], 
				1);
			cur_fc_input = [conv_out];
			fc_intermediate = [];

			for i in range(len(fc_b)):
				fc_intermediate.append(
					tf.matmul(
					tf.nn.dropout(cur_fc_input[i], keep_prob = dropout_keep_prob), fc_W[i]
					) + fc_b[i]
					);
				if i < len(fc_b) - 1:
					cur_fc_input.append(lrelu(bn_layer(
												fc_intermediate[i],
												is_training = is_training,
												decay = hp.Model["BATCH_NORM"]["DECAY"],
												epsilon = hp.Model["BATCH_NORM"]["EPS"]
												)
										)
					);
				else:
					cur_fc_input.append(fc_intermediate[i]);

			return cur_fc_input[len(fc_b)];

	def wnorm(self):
		return tf.nn.l2_loss(self.parameters["conv_W"][0]);

	def loss(self,pred_batch, label_batch):
		l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_batch, logits = pred_batch));
		for w in self.parameters["conv_W"]:
			l = l + hp.Model["REG_RATE"] * tf.nn.l2_loss(w) / self.num_parameters;

		for w in self.parameters["fc_W"]:
			l = l + hp.Model["REG_RATE"] * tf.nn.l2_loss(w) / self.num_parameters;

		return l;

	def train(self):
		train_pred = self.network(self.train_data, 
								  is_training = True 
								  );
		train_labels = (tf.cast(tf.reshape(self.train_label, [-1,1]), tf.float32) + 1) / 2;
		loss = self.loss(train_pred, train_labels);
		_wnorm = self.wnorm();
		val_accuracy = None;
		val_pred = None;
		val_dropout_keep_prob = None;
		val_is_training = None;
		val_labels = None;
		num_correct = None;
		val_loss = None;
		val_accuracies = [];
		if self.using_val:
			val_pred = self.network(self.val_data, is_training = False);
			val_labels = (tf.cast(tf.reshape(self.val_label, [-1, 1]), tf.float32) + 1) / 2;
			num_correct = tf.reduce_sum(
				tf.cast(tf.logical_and(val_pred > 0.5, val_labels > 0.6), tf.float32) + 
				tf.cast(tf.logical_and(val_pred < 0.5, val_labels < 0.4), tf.float32)
			);
			val_accuracy = num_correct / \
				(tf.reduce_sum(
					tf.cast(val_labels > 0.6, tf.float32) + 
					tf.cast(val_labels < 0.4, tf.float32)
				));
			val_loss = self.loss(val_pred, val_labels);

		with tf.Session() as sess:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
			with tf.control_dependencies(update_ops):
				train_step = tf.train.AdamOptimizer(hp.Model["LEARNING_RATE"]).minimize(loss);
			sess.run(tf.global_variables_initializer());
			coord = tf.train.Coordinator();
			threads = tf.train.start_queue_runners(coord = coord);
			losses = [];
			for i in range(hp.Model["NUM_RUN"]):
				_,loss_,_val_loss = sess.run([train_step, loss, val_loss]);
				losses.append(loss_);
				if i%50 == 0:
					print _val_loss;
					print loss_;
					print i;
					if self.using_val:
						val_accuracy_, num_correct_ = sess.run([val_accuracy, num_correct]);
						val_accuracies.append(val_accuracy_);

			coord.request_stop();
			coord.join(threads);
		return losses, val_accuracies;

#################################################################

if __name__ == "__main__":
	board_batch, label_batch = read_dataset(filenames = [TFRECORD_FOLDER + str(i) + ".tfrecord" for i in range(40)]);
	val_board_batch, val_label_batch = read_dataset(filenames = [TFRECORD_FOLDER + str(i) + ".tfrecord" for i in range(40,44)]);
	agent = Agent(train_data = board_batch, train_label = label_batch, 
		using_val = True, val_data = val_board_batch, val_label = val_label_batch);
	losses, val_accuracies = agent.train();

	plt.plot(losses[100:]);
	plt.figure();
	plt.plot(val_accuracies);
	plt.show();
	plt.show('hold');
