Input={
	"BATCH_SIZE":			512,
	"NUM_THREADS":			4,
	"MIN_AFTER_DEQUEUE":	200,
	"CAPACITY":				0
}

Input["CAPACITY"] = Input["MIN_AFTER_DEQUEUE"]+Input["BATCH_SIZE"]*3;

Model={
	"WEIGHT_INIT_STD":		0.1,
	"DROP_OUT_KEEP_PROB":	0.95,
	"REG_RATE":				5e-2,
	"LEARNING_RATE":		5e-6,
	"NUM_RUN":				1000,
	"RELU_LEAK_ALPHA":		0.1,
	"BATCH_NORM":			{
		"DECAY":			0.999,
		"EPS":				1e-3,
		"RENORM":			False,
		"RENORM_DECAY":		0.999
	}
}