Input={
	"BATCH_SIZE":			512,
	"NUM_THREADS":			2,
	"MIN_AFTER_DEQUEUE":	200,
	"CAPACITY":				0
}

Input["CAPACITY"] = Input["MIN_AFTER_DEQUEUE"]+Input["BATCH_SIZE"]*5;

Model={
	"WEIGHT_INIT_STD":		0.5,
	"DROP_OUT_KEEP_PROB":	0.95,
	"REG_RATE":				0.5,
	"LEARNING_RATE":		1e-2,
	"NUM_RUN":				5,
	"RELU_LEAK_ALPHA":		0.1,
	"BATCH_NORM":			{
		"TRAIN_BN_NAME":	"bn_train",
		"DECAY":			0.999,
		"EPS":				1e-3,
		"RENORM":			True,
		"RENORM_DECAY":		0.999
	}
}