import numpy as np;
import tensorflow as tf;
from sets import Set;
from global_names import *;

piece = {'K':1,'Q':2,'R':3,'N':4,'B':5,'P':6, 'PP':7};
numToPieceStr = {0:' ',1:'K',2:'Q',3:'R',4:'N',5:'B',6:'P',7:'.'};
WHITE_WIN = 1;
BLACK_WIN = -1;
TILE = 0;

def stringToPos(s):
	return ((ord(s[1])-ord('1')),(ord(s[0])-ord('a')));

class ChessGame:
	def __init__(self):
		self.curPlaying=1;
		xString = ['R','N','B','Q','K','B','N','R'];
		x = [piece[s] for s in xString];
		xx = [piece['P'] for i in range(8)];

		empty = [0 for i in range(8)];
		boardList = [x[:],xx[:]];
		for i in range(4):
			boardList.append(empty[:]);
		boardList.append(xx[:]);
		boardList.append(x[:]);
		
		self.board=np.matrix(boardList);
		self.board[6:8,:]=self.board[6:8,:]*(-1);

		firstLine = lambda color: str(1) if color==1 else str(8);
		secondLine = lambda color: str(2) if color==1 else str(7);

		poses = lambda color:\
				{	piece['R']:Set([stringToPos("a"+firstLine(color)),stringToPos("h"+firstLine(color))]),\
						piece['N']:Set([stringToPos("b"+firstLine(color)),stringToPos("g"+firstLine(color))]),\
						piece['B']:Set([stringToPos("c"+firstLine(color)),stringToPos("f"+firstLine(color))]),\
						piece['Q']:Set([stringToPos("d"+firstLine(color))]),\
						piece['K']:Set([stringToPos("e"+firstLine(color))]),\
						piece['P']:Set([stringToPos(chr(i)+secondLine(color) ) for i in range(ord('a'),ord('i'))]),\
						piece['PP']:Set([])
					};

		self.piecePoses={1:poses(1),
								-1:poses(-1)};
		self.OOValid = True;
		self.OOOValid = True;
		self.win = 2;
	
	def getPieceAtString(self,s):
		return self.getPieceAtPos(*stringToPos(s));

	def getPieceAtPos(self,x,y):
		return self.board[x,y];

	def __str__(self):
		out = "";
		for i in range(7,-1,-1):
			out = out+'|';
			for j in range(8):
				curStr=numToPieceStr[abs(self.board[i,j])];
				if(self.board[i,j]<0):
					curStr=curStr+'*';
				else:
					curStr=curStr+' ';
				out = out+curStr+'|';
			out = out+"\n";
		return out;

	def updateValidMove(self,x,y,targetX,targetY,curPiece,curColor,moves,noCheckDirectionFlag=False):
		if targetX<0 or targetX>7 or targetY<0 or targetY>7:
			return False;

		tarPiece = self.getPieceAtPos(targetX,targetY);
		if abs(curPiece)==piece["P"]:
			if tarPiece==0 and y==targetY:
				if not self.validMoveResultInCheck(x,y,targetX,targetY,curPiece,curColor):
					moves.add((targetX,targetY));
				return True;

			if y==targetY:
				return False;

			if tarPiece!=0 and (tarPiece>0)!=curColor:
				if not self.validMoveResultInCheck(x,y,targetX,targetY,curPiece,curColor):
					moves.add((targetX,targetY));

			return True;


		if tarPiece==0 or abs(tarPiece)==piece['PP']:
			if not self.validMoveResultInCheck(x,y,targetX,targetY,curPiece,curColor):
				moves.add((targetX,targetY));
			return True;

		if (tarPiece>0)!=curColor:#different color
			if not self.validMoveResultInCheck(x,y,targetX,targetY,curPiece,curColor):
				moves.add((targetX,targetY));

		return False;


	def posHasOpponent(self,x,y,curColor):
		if x<0 or x>7 or y<0 or y>7:
		#invalid
			return -2;
		tarPiece = self.getPieceAtPos(x,y);
		if tarPiece==0 or abs(tarPiece)==piece['PP']:
		#empty
			return -1;
		if (tarPiece>0)==curColor:
		#self
			return 0;
		return abs(tarPiece);

	def validMoveResultInCheck(self, x,y,targetX,targetY,curPiece,curColor):
		if targetX<0 or targetX>7 or targetY<0 or targetY>7 or x<0 or x>7 or y<0 or y>7:
			return False;

		targetPiece = self.getPieceAtPos(targetX,targetY);

		self.board[x,y]=0;
		self.board[targetX,targetY]=curPiece;
		enPassant = abs(targetPiece)==piece["PP"] and abs(curPiece)==piece["P"];
		if enPassant:
			self.board[x,targetY]=0;

		kingPos = (targetX,targetY);
		if abs(curPiece)!=piece["K"]:
			for pos in self.piecePoses[1 if curColor else -1][piece["K"]]:
				kingPos = pos;
				break;

		if self.posAttacked(kingPos[0],kingPos[1],curColor):
			self.board[x,y]=curPiece;
			self.board[targetX,targetY]=targetPiece;
			if enPassant:
				self.board[x,targetY]=piece["P"] if not curColor else 0-piece["P"];
			return True;

		self.board[x,y]=curPiece;
		self.board[targetX,targetY]=targetPiece;
		if enPassant:
			self.board[x,targetY]=piece["P"] if not curColor else 0-piece["P"];
		return False;

	def posAttacked(self,x,y,curColor):
		for direction in range(4):
			a = direction/2;
			b = direction%2;
			for i in range(1,8):
				d = i if b==0 else 0-i;
				dx = 0 if a==0 else d;
				dy = d if a==0 else 0;
				newPosFlag = self.posHasOpponent(x+dx,y+dy,curColor);
				if newPosFlag>0:
					if newPosFlag==piece['R'] or newPosFlag==piece['Q']:
						return True;
					break;
				if newPosFlag!=-1:
					break;

			for i in range(1,8):
				dx = i if a==0 else 0-i;
				dy = i if b==0 else 0-i;
				newPosFlag = self.posHasOpponent(x+dx,y+dy,curColor);
				if newPosFlag>0:
					if newPosFlag==piece['B'] or newPosFlag==piece['Q']:
						return True;
					break;
				if newPosFlag!=-1:
					break;

			for i in [1,-1]:
				dx = 1 if a==0 else -1;
				dy = 1 if b==0 else -1;
				dx = dx*(2 if i==1 else 1);
				dy = dy*(1 if i==1 else 2);
				newPosFlag = self.posHasOpponent(x+dx,y+dy,curColor);
				if newPosFlag==piece['N']:
					return True;

		#pawns
		dx = 1 if curColor else -1;
		for dy in [-1,1]:
			newPosFlag = self.posHasOpponent(x+dx,y+dy,curColor);
			if newPosFlag==piece['P']:
				return True;

		#king
		for dx in range(-1,2):
			for dy in range(-1,2):
				if not (dx==0 and dy==0):
					newPosFlag = self.posHasOpponent(x+dx,y+dy,curColor);
					if newPosFlag==piece['K']:
						return True;

		return False;

	def getValidMoves(self,x,y):
		curMoves = Set([]);
		curPiece = self.getPieceAtPos(x,y);

		if curPiece==0:
			return curMoves;

		for direction in range(4):
			a = direction/2;
			b = direction%2;
			if abs(curPiece)==piece['R'] or abs(curPiece)==piece['Q']:
				d = 1 if b==0 else -1;
				dx = 0 if a==0 else d;
				dy = d if a==0 else 0;
				for i in range(1,8):
					if not self.updateValidMove(x,y,x+i*dx,y+i*dy,curPiece,curPiece>0,curMoves):
						break;

			if abs(curPiece)==piece['B'] or abs(curPiece)==piece['Q']:
				dx = 1 if a==0 else -1;
				dy = 1 if b==0 else -1;
				for i in range(1,8):
					if not self.updateValidMove(x,y,x+i*dx,y+i*dy,curPiece,curPiece>0,curMoves):
						break;

		if abs(curPiece)==piece['N']:
			for i in range(8):
				a = i/4;
				b = (i%4)/2;
				c = i%2;
				dx = (-1 if b==1 else 1)*(1 if a==1 else 2);
				dy = (-1 if c==1 else 1)*(2 if a==1 else 1);
				self.updateValidMove(x,y,x+dx,y+dy,curPiece,curPiece>0,curMoves);

		if abs(curPiece)==piece['P']:
			dx = 1;
			if curPiece<0:
				dx = -1;
			if self.updateValidMove(x,y,x+dx,y,curPiece,curPiece>0,curMoves):
				if x==(1 if curPiece>0 else 6):
					self.updateValidMove(x,y,x+dx+dx,y,curPiece,curPiece>0,curMoves);

			for dy in [1,-1]:
				if not (y+dy<0 or y+dy>7):
					self.updateValidMove(x,y,x+dx,y+dy,curPiece,curPiece>0,curMoves);

		if abs(curPiece)==piece['K']:
			for dx in range(-1,2):
				for dy in range(-1,2):
					if not (dx==0 and dy==0):
						self.updateValidMove(x,y,x+dx,y+dy,curPiece,curPiece>0,curMoves);
			if self.OOValid:
				curMoves.add((-1,-1));

			if self.OOOValid:
				curMoves.add((-2,-2));

		return curMoves;

	def toNextState(self,s):
		if "1-0" in s:
			self.win = WHITE_WIN;
			return;

		if "0-1" in s:
			self.win = BLACK_WIN;
			return;

		if "1/2-1/2" in s:
			self.win = TILE;
			return;

		if "O-O" in s and "O-O-O" not in s:
			x = 0 if self.curPlaying==1 else 7;
			self.board[x,4]=0;
			self.board[x,5]=self.curPlaying*piece['R'];
			self.board[x,6]=self.curPlaying*piece['K'];
			self.board[x,7]=0;

			self.piecePoses[self.curPlaying][piece['R']].remove((x,7));
			self.piecePoses[self.curPlaying][piece['R']].add((x,5));
			self.piecePoses[self.curPlaying][piece['K']].remove((x,4));
			self.piecePoses[self.curPlaying][piece['K']].add((x,6));
			OOValid = False;
			OOOValid = False;
			for pos in self.piecePoses[0-self.curPlaying][piece["PP"]]:
				self.board[pos[0],pos[1]]=0;

			self.piecePoses[0-self.curPlaying][piece["PP"]].clear();
			self.curPlaying=0-self.curPlaying;

			return;

		if "O-O-O" in s:
			x = 0 if self.curPlaying==1 else 7;
			self.board[x,4]=0;
			self.board[x,3]=self.curPlaying*piece['R'];
			self.board[x,2]=self.curPlaying*piece['K'];
			self.board[x,0]=0;

			self.piecePoses[self.curPlaying][piece['R']].remove((x,0));
			self.piecePoses[self.curPlaying][piece['R']].add((x,3));
			self.piecePoses[self.curPlaying][piece['K']].remove((x,4));
			self.piecePoses[self.curPlaying][piece['K']].add((x,2));
			OOValid = False;
			OOOValid = False;
			for pos in self.piecePoses[0-self.curPlaying][piece["PP"]]:
				self.board[pos[0],pos[1]]=0;

			self.piecePoses[0-self.curPlaying][piece["PP"]].clear();
			self.curPlaying=0-self.curPlaying;
			return;

		curPiece = piece['P'];
		if ord(s[0])<ord('a'):
			curPiece = piece[s[0]];

		sourceY = -1;
		sourceX = -1;
		targetY = 0;
		targetX = 0;
		promotePiece = 0;
		targetRecorded=False;
		for j in range(len(s)):
			index = len(s)-1-j;
			if ord('A')<=ord(s[index]) and ord(s[index])<=ord('Z'):
				promotePiece=piece[s[index]];

			if not targetRecorded:
				if ord('a')<=ord(s[index]) and ord(s[index])<=ord('h'):
					if ord('1')<=ord(s[index+1]) and ord(s[index+1])<=ord('8'):
						targetRecorded=True;
						targetY=ord(s[index])-ord('a');
						targetX=ord(s[index+1])-ord('1');
			else:
				if ord('a')<=ord(s[index]) and ord(s[index])<=ord('h'):
					sourceY=ord(s[index])-ord('a');

				if ord('1')<=ord(s[index]) and ord(s[index])<=ord('8'):
					sourceX=ord(s[index])-ord('1');

		if sourceY<0 and sourceX<0:
			for pos in self.piecePoses[self.curPlaying][curPiece]:
				if (targetX,targetY) in self.getValidMoves(*pos):
					sourceX=pos[0];
					sourceY=pos[1];
					break;
		elif sourceY<0 or sourceX<0:
			for pos in self.piecePoses[self.curPlaying][curPiece]:
				if (targetX,targetY) in self.getValidMoves(*pos) and (sourceX==pos[0] or sourceY==pos[1]):
					sourceX=pos[0];
					sourceY=pos[1];
					break;

		targetPiece = abs(self.getPieceAtPos(targetX,targetY));
		if targetPiece==piece["PP"] and curPiece==piece["P"]:
			self.piecePoses[0-self.curPlaying][piece["P"]].remove((sourceX,targetY));
			self.board[sourceX,targetY]=0;

		elif targetPiece>0:
			self.piecePoses[0-self.curPlaying][targetPiece].remove((targetX,targetY));

		for pos in self.piecePoses[0-self.curPlaying][piece["PP"]]:
			self.board[pos[0],pos[1]]=0;

		self.piecePoses[0-self.curPlaying][piece["PP"]].clear();

		self.piecePoses[self.curPlaying][curPiece].remove((sourceX,sourceY));
		self.board[sourceX,sourceY]=0;

		if curPiece==piece["P"] and promotePiece>0:
			self.piecePoses[self.curPlaying][promotePiece].add((targetX,targetY));
			self.board[targetX,targetY] = self.curPlaying*promotePiece;

		else:
			self.piecePoses[self.curPlaying][curPiece].add((targetX,targetY));
			self.board[targetX,targetY] = self.curPlaying*curPiece;
		
		if curPiece==piece["P"] and abs(targetX - sourceX)==2:
			self.board[sourceX+self.curPlaying,targetY]=self.curPlaying*piece["PP"];
			self.piecePoses[self.curPlaying][piece["PP"]].add((sourceX+self.curPlaying,targetY));

		self.curPlaying=0-self.curPlaying;

if __name__=="__main__":
	game_file = open(DATA_ROOT+"game_0.txt",'r');
	count = 0;

	"""for line in game_file:
		print count;
		steps=line.split(" ")[1:];
		test_file = open("test/test.out",'w');
		c=ChessGame();
		for i in range(len(steps)):
			if not i%3==0:
				c.toNextState(steps[i]);
				test_file.write(steps[i]+"\n");
				test_file.write(str(c));
				test_file.write("----------------------------------------------\n");
		test_file.close();
		count = count+1;
	game_file.close();

	"""
	game_file = open(DATA_ROOT+"game_158.txt",'r');
	count = 0;

	line = game_file.readlines()[5265];
	steps=line.split(" ")[1:];
	test_file = open("test/test.out",'w');
	c=ChessGame();
	for i in range(len(steps)):
		if not i%3==0:
			print steps[i];
			c.toNextState(steps[i]);
			test_file.write(steps[i]+"\n");
			test_file.write(str(c));
			test_file.write("----------------------------------------------\n");
			#if "Nxb4" in steps[i]:
				#break;
	test_file.close();
	count = count+1;
	game_file.close();#"""


