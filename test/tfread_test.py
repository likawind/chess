import tensorflow as tf;
import numpy as np;

record_iterator = tf.python_io.tf_record_iterator(path="test.tfrecord");
for record in record_iterator:
	example = tf.train.Example();
	example.ParseFromString(record);

	label = example.features.feature['label'].int64_list.value[0];
	print label;

	state = example.features.feature['state'].bytes_list.value[0];
	state = np.fromstring(state,dtype=int).reshape(8,8);
	print state;