import tensorflow as tf;
import numpy as np;
import hyper_parameters;
import sys;
from global_names import *;

def board2dTo3d(board2d):

	indices_pos = tf.where(board2d>0);
	player_pieces = tf.concat([indices_pos,tf.reshape(tf.gather_nd(board2d,indices_pos)-1,[-1,1])],1);
	indices_neg = tf.where(board2d<0);
	opponent_pieces = tf.concat([indices_neg,tf.reshape(tf.gather_nd(tf.abs(board2d),indices_neg)-1,[-1,1])],1);

	board3d = tf.scatter_nd(tf.concat([player_pieces,opponent_pieces],0),
					tf.concat([tf.ones([tf.shape(indices_pos)[0]]),
							-1*tf.ones([tf.shape(indices_neg)[0]])],0),
					tf.cast(tf.constant([8,8,7]),tf.int64));

	return board3d;

def read_tfrecord(filename_queue):
	reader = tf.TFRecordReader();
	_, single_example = reader.read(filename_queue);
	parsed_example = tf.parse_single_example(
		single_example,
		features = {
			"state": tf.FixedLenFeature([],tf.string),
			"label": tf.FixedLenFeature([],tf.int64)
		}
		);

	state_string = tf.decode_raw(parsed_example["state"],tf.int64);
	board2d = tf.reshape(state_string,[8,8]);
	label = tf.cast(parsed_example["label"],tf.int64);

	board3d = board2dTo3d(board2d);
	return board3d, label;

def read_dataset(filenames):
	filename_queue = tf.train.string_input_producer(
		filenames, num_epochs=None, shuffle=True);
	board3d, label = read_tfrecord(filename_queue);
	board3d_batch, label_batch = tf.train.shuffle_batch(
		[board3d, label],
		batch_size 			= hyper_parameters.Input["BATCH_SIZE"],
		num_threads 		= hyper_parameters.Input["NUM_THREADS"],
		capacity			= hyper_parameters.Input["CAPACITY"],
		min_after_dequeue	= hyper_parameters.Input["MIN_AFTER_DEQUEUE"]
		);

	return board3d_batch, label_batch;

if __name__ == "__main__":
	board3d_batch, label_batch = read_dataset(filenames=[TFRECORD_FOLDER+str(i)+".tfrecord" for i in range(10)]);
	
	with tf.Session() as sess:
		coord = tf.train.Coordinator();
		threads = tf.train.start_queue_runners(coord=coord);
		log_file_board = open("test/read_test_board.out","w");
		log_file_label = open("test/read_test_label.out","w");
		for i in range(1):
			sess.run([board3d_batch, label_batch]);
			print "----------------";
			#np.savetxt(log_file_board, board_batch[0].eval(),fmt="%d");
			#np.savetxt(log_file_label, label_batch.eval(),fmt="%d");
			

		log_file_board.close();
		log_file_label.close();
		coord.request_stop();
		coord.join(threads);

"""
INPUT_DIM = 32768;
OUTPUT_DIM = 2;

WEIGHT_INIT_STDDEV = 0.1;
x = tf.placeholder(tf.float32, shape=[None,INPUT_DIM]);
y_train = tf.placeholder(tf.float32, shape=[None,OUTPUT_DIM]);

x_1 = tf.reshape(x,[-1,32,32,32,1]);
W_1 = weight([3,3,3,32]);"""