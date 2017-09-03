from global_names import *;

num_games_per_file = 10000;

all_games_pgn_file_name = DATA_ROOT+"millionbase-2.22.pgn";
all_games_pgn = open(all_games_pgn_file_name,'r');

gameCount = 0;
fileCount = 0;
curGame = "";
newGame = 1;
cur_game_file_name=DATA_ROOT+"game_"+str(fileCount)+".txt";
cur_game_file = open(cur_game_file_name,'w');
for line in all_games_pgn:
	if '[' not in line and '.' in line:
		curGame = curGame+" "+line[:-2];
		newGame = 0;
	elif '[' in line and newGame==0:
		cur_game_file.write(curGame+"\n");
		curGame = "";
		newGame = 1;
		gameCount += 1;
		if gameCount>=num_games_per_file:
			print gameCount,fileCount;
			gameCount = 0;
			fileCount += 1;
			cur_game_file.close();
			cur_game_file_name=DATA_ROOT+"game_"+str(fileCount)+".txt";
			cur_game_file = open(cur_game_file_name,'w');

all_games_pgn.close();
