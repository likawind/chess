import tensorflow as tf;
import numpy as np;
import hyper_parameters;

def board2dTo3d(board2d):
	return board3d;

def read_tfrecord(filename_queue):
	reader = tf.TFRecordReader();
	_, single_example = reader.read(filename_queue);
	parsed_example = tf.parse_single_example(
		single_example,
		features = {
			"board": tf.FixedLenFeature([],tf.string);
			"label": tf.FixedLenFeature([],tf.int64);
		}
		);

	board2d = ;
	label = ;

	board3d = board2dTo3d(board2d);
	return board3d, label;

def read_dataset(filenames, batch_size):
	filename_queue = tf.train.string_input_producer(
		filenames, num_epochs=None, shuffle=True);
	board, label = read_tfrecord(filename_queue);
	board_batch, label_batch = tf.train.shuffle_batch(
		[board, label],
		batch_size 			= hyper_parameters.Input["BATCH_SIZE"],
		num_threads 		= hyper_parameters.Input["NUM_THREADS"],
		capacity			= hyper_parameters.Input["CAPACITY"],
		min_after_dequeue	= hyper_parameters.Input["MIN_AFTER_DEQUEUE"]
		);
	return board_batch, label_batch;

"""
INPUT_DIM = 32768;
OUTPUT_DIM = 2;

WEIGHT_INIT_STDDEV = 0.1;
x = tf.placeholder(tf.float32, shape=[None,INPUT_DIM]);
y_train = tf.placeholder(tf.float32, shape=[None,OUTPUT_DIM]);

x_1 = tf.reshape(x,[-1,32,32,32,1]);
W_1 = weight([3,3,3,32]);"""