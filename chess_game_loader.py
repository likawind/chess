from chess_game import *;
from global_names import *;

import threading;
import tensorflow as tf;

def toSparseMatrix(game):
	sparseMatrix = np.zeros((8,8,7));
	sparseMatrix[np.ndarray.tolist(np.where(game.board>0)[0]),np.ndarray.tolist(np.where(game.board>0)[1]),\
		np.ndarray.tolist(game.board[game.board>0]-1)]=0-game.curPlaying;
	sparseMatrix[np.ndarray.tolist(np.where(game.board<0)[0]),np.ndarray.tolist(np.where(game.board<0)[1]),\
		np.ndarray.tolist(abs(game.board[game.board<0])-1)]=game.curPlaying;
	return sparseMatrix;

class ChessGameLoader:
	def __init__(self, game_file_name_list, record_file_name):
		self.game_file_name_list = game_file_name_list;
		self.record_file_name = record_file_name;

	def generateDatasetMatrix(self):
		writer = tf.python_io.TFRecordWriter(self.record_file_name);

		for game_file_name in self.game_file_name_list:
			game_file = open(game_file_name,'r');
			count = 0;
			abandon_game = False;
			for line in game_file:
				if count%100 == 0:
					print game_file_name+" "+str(count);
				steps=line.split(" ")[1:];
				states = [];
				count += 1;
				game = ChessGame();

				for i in range(len(steps)):
					if "." not in steps[i]:
						if "*" in steps[i]:
							abandon_game = True;
							break;
						#states.append(toSparseMatrix(game).astype(int));
						states.append((game.board*game.curPlaying).copy().astype(int));
						try:
							game.toNextState(steps[i]);
						except:
							abandon_game = True;
							log_file = open(STORING_LOG_FILE,"a");
							log_file.write(game_file_name+"\n"+line+"\n"+steps[i]+"\n----------------------------");
							log_file.close();
							break;

				if abandon_game:
					abandon_game = False;
				else:
					if game.win==TILE:
						labels = np.zeros(len(states));
					else:
						labels = np.ones(len(states));
						if game.win == BLACK_WIN:
							labels[0:len(states):2]=-1;
						else:
							labels[1:len(states):2]=-1;

					labels = np.ndarray.tolist(labels.astype(int));
					for i in range(len(states)):
						example = tf.train.Example(features=tf.train.Features(feature={
							"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
							"state": tf.train.Feature(bytes_list=tf.train.BytesList(value=[states[i].tobytes()]))
						}));
						writer.write(example.SerializeToString());

			game_file.close();

		writer.close();

class StoringThread(threading.Thread):
	def __init__(self, threadIndex):
		threading.Thread.__init__(self);
		self.index = threadIndex;
		
	def run(self):
		global curFileCount;
		global tfrecordCount;
		log_file = open("storingToTFRecordLog"+str(self.index)+".log", "w");
		while(1):
			threadLock.acquire();
			if(curFileCount>= NUM_RAW_CHESS_DATA_FILE):
				threadLock.release();
				return;
			file_list = [DATA_ROOT+"game_"+str(i+curFileCount)+".txt" for i in range(NUM_RAW_DATA_FILE_PER_TFRECORD)];
			target_tfrecord_file = TFRECORD_FOLDER+str(tfrecordCount)+".tfrecord";
			curFileCount += NUM_RAW_DATA_FILE_PER_TFRECORD;
			tfrecordCount += 1;
			print file_list;
			print target_tfrecord_file;
			threadLock.release();
			try:
				loader = ChessGameLoader(file_list,target_tfrecord_file);
				loader.generateDatasetMatrix();
			except:
				log_file.write(file_list);

		log_file.close();
	

if __name__=="__main__":
	curFileCount = 0;
	tfrecordCount = 0;
	threadLock = threading.Lock();
	threads = [StoringThread(i) for i in range(4)];

	for t in threads:
		t.start();

	for t in threads:
		t.join();


