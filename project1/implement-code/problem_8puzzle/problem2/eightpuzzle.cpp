#include "eightpuzzle.hpp"
#include <vector>
#include <queue>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <time.h>
using std::random_shuffle;
using namespace std;
/* Your A* search implementation for 8 puzzle problem
NOTICE:
1. iniState.state is a 3x3 matrix, the "space" is indicated as -1, for example
	1 2 -1              1 2
	3 4 5   stands for  3 4 5
	6 7 8               6 7 8
2. moves contains directions that transform initial state to final state. Here
	0 stands for up
	1 stands for down
	2 stands for left
	3 stands for right
   There might be several ways to understand "moving up/down/left/right". Here we define
   that "moving up" means to move other numbers up, not move "space" up. For example
   	1 2 -1              1 2 5
	3 4 5   move up =>  3 4 -1
	6 7 8               6 7 8
   This definition is actually consistent with where your finger moves to
    when you are playing 8 puzzle game.
    Your code start here
*/
void reconstruct_path(EightPuzzleState* current, EightPuzzleState& iniState, vector<int>& moves)
{
	EightPuzzleState* p = current;
	while(*p != iniState)
	{
		moves.push_back(p->preMove);
		p = p ->preState;
	}
	int number = moves.size();

	//for (int i = 0; i < number; i++)
	//{
	//	printf("%d   ", moves[i]);
	//}
	moves.reserve(moves.size());

}

void getSuccessors(EightPuzzleState* current, vector<EightPuzzleState*>& successors)
{
	EightPuzzleState * next_state = new EightPuzzleState[4];

	for (int i =0; i < 4; ++i)
	{
		runOneMove(current, next_state + i, i);
		//printState(next_state + i);

		successors.push_back(next_state + i);
	}
	
}

int computeHeuristic(EightPuzzleState& current)
{
    int i = 0, j = 0;
	const int finalState[3][3] = {{1,2,3},{4,5,6},{7,8,-1}};
	int k = 0;
	for (i = 0 ; i < 3 ;i++)
		for(j = 0; j < 3 ; j++)
		{
			if(current.state[i][j] != finalState[i][j])
				k++;
		}
	return k;
}


bool findOpenlist(priority_queue<EightPuzzleState*, vector<EightPuzzleState*>, cmpLarge> openlist, EightPuzzleState * Node)
{
	vector<EightPuzzleState*> member;
	bool flag = false;
	while(! openlist.empty())
	{
		EightPuzzleState* temp = openlist.top();
		openlist.pop();
		member.push_back(temp);
		if(* temp == * Node)
		{
			flag = true;
			int number = member.size();
			for (int i = 0; i < number; i++)
			{
				openlist.push(member[i]);
			}
			break;
		}
	}

	if (flag == false)
	{
		int number = member.size();
		for (int i = 0; i < number; i++)
		{
			openlist.push(member[i]);
		}
	}

	return flag;
}


bool findCloselist(vector<EightPuzzleState*> closelist, EightPuzzleState * Node)
{
	bool flag = false;
	int number = closelist.size();
	for (int i = 0; i < number; i++)
	{
		if(* closelist[i] == * Node)
		{
			flag = true;
			break;
		}
	}

	return flag;
}


void AStarSearchFor8Puzzle(EightPuzzleState& iniState, vector<int>& moves)
{
	priority_queue<EightPuzzleState*, vector<EightPuzzleState*>, cmpLarge> openlist;
	vector<EightPuzzleState*> closelist;
	openlist.push(&iniState);
	while(! openlist.empty())
	{
         EightPuzzleState* current = openlist.top();
		 if(checkFinalState(current))
			 return reconstruct_path(current, iniState, moves);

		 openlist.pop();
		 closelist.push_back(current);

		 vector<EightPuzzleState*>  successors;
		 getSuccessors(current, successors);

		 for (int i = 0 ; i < (int)successors.size() ; ++i)
		 {
			 bool result1 = findOpenlist(openlist, successors[i]);
			 bool result2 = findCloselist(closelist, successors[i]);

			 if (result2)
			 {
				 continue;
			 }
			 if (!result1)
			 {
				 successors[i]->preState = current;
				 successors[i]->h = computeHeuristic(*successors[i]);
				 successors[i]->g = current->g + 1;
				 successors[i]->preMove = i;
				 // printState(successors[i]);
				 openlist.push(successors[i]);
			 }
			 else if(current->g + 1 >= successors[i]->g)
				 continue;
			 successors[i]->preState = current;
			 successors[i]->h = computeHeuristic(*successors[i]);
			 successors[i]->g = current->g + 1;
			 successors[i]->preMove = i;

		 }
	}

}

// You may need the following code, but you may not revise it

/* Play 8 puzzle game according to "moves" and translate state from "iniState" to  "resultState"
   This function is used to check your AStarSearchFor8Puzzle implementation.
   It will return a state to indicate whether the vector moves will lead the final state.
   return 1 means moves are correct!
   return -1 means moves can not turn iniState to final state
   return -2 means moves violate game rule, see runOneMove();
   You should not revise this function.
*/
int runMoves(const EightPuzzleState* iniState, const vector<int>& moves)
{
	if (moves.size() == 0)
	{
		return -1;
	}
	//memcpy(&resultState[0][0], &iniState[0][0], sizeof(*iniState));
	EightPuzzleState currentState = *iniState;
	EightPuzzleState nextState;
	for (int i = 0; i < (int)moves.size(); ++i)
	{
		if (!runOneMove(&currentState, &nextState, moves[i]))
		{
			return -2;
		}
		currentState = nextState;
	}
	if (checkFinalState(&nextState))
	{
		return 1;
	}
	else
	{
		return -1;
	}
}

bool checkMove(const EightPuzzleState* state, const int move,
	int& r_1, int& c_1, int& r_1_move, int& c_1_move)
{
	for (int r = 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
		{
			if (state->state[r][c] == -1)
			{
				r_1 = r;
				c_1 = c;
			}
		}
	}

	switch (move)
	{
		//up
	case 0:
		r_1_move = r_1 + 1;
		c_1_move = c_1;
		break;
		//down
	case 1:
		r_1_move = r_1 - 1;
		c_1_move = c_1;
		break;
		//left
	case 2:
		c_1_move = c_1 + 1;
		r_1_move = r_1;
		break;
		//right
	case 3:
		c_1_move = c_1 - 1;
		r_1_move = r_1;
	}

	// if move out of boundary
	if (r_1_move < 0 || r_1_move > 2 || c_1_move < 0 || c_1_move > 2)
	{
		return false;
	}
	return true;
}

bool runOneMove(EightPuzzleState* preState, EightPuzzleState* nextState, const int move)
{
	// find the position of -1
	int r_1, c_1, r_1_move, c_1_move;
	*nextState = *preState;
	bool flag = checkMove(nextState, move, r_1, c_1, r_1_move, c_1_move);

	// if move out of boundary
	if (r_1_move < 0 || r_1_move > 2 || c_1_move < 0 || c_1_move > 2)
	{
		return false;
	}
	int v = nextState->state[r_1_move][c_1_move];
	nextState->state[r_1][c_1] = v;
	nextState->state[r_1_move][c_1_move] = -1;
	nextState->preState = preState;
	nextState->preMove = move;
	return true;
}

bool checkFinalState(const EightPuzzleState* resultState)
{
	const int finalState[3][3] = {{1,2,3},{4,5,6},{7,8,-1}};

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (finalState[i][j] != resultState->state[i][j])
			{
				return false;
			}
		}
	}
	return true;
}

void generateState(EightPuzzleState* state, int nMoves)
{
	EightPuzzleState finalState;
	for (int i =1; i < 10; ++i)
	{
		finalState.state[(i-1)/3][(i-1)%3] = i;
	}
	finalState.state[2][2] = -1;
	EightPuzzleState preState, nextState;
	preState = finalState;
	srand((int)time(0));
	for (int i =0; i < nMoves; ++i)
	{
		int rdmove = rand()%4;
		runOneMove(&preState, &nextState, rdmove);
		preState = nextState;
	}
	//²âÊÔÊý¾Ý
	//int count = 1;
	//for (int i = 0; i < 2; i++)
	//for (int j = 0; j < 3; j++)
	//{
	//	nextState.state[i][j] = count++;
	//}
	//nextState.state[2][0] = -1;
	//nextState.state[2][1] = 7;
	//nextState.state[2][2] = 8;
	*state = nextState;
}

void printMoves(EightPuzzleState* state, vector<int>& moves)
{
	cout << "Initial state " << endl;
	printState(state);
	EightPuzzleState preState, nextState;
	preState = *state;
	for (int i =0; i < (int)moves.size(); ++i)
	{
		switch (moves[i])
		{
		case 0:
			cout << " The " << i << "-th move goes up" << endl;
			break;
		case 1:
			cout << " The " << i << "-th move goes down" << endl;
			break;
		case 2:
			cout << " The " << i << "-th move goes left" << endl;
			break;
		case 3:
			cout << " The " << i << "-th move goes right" << endl;
		}

		runOneMove(&preState, &nextState, moves[i]);
		printState(&nextState);
		preState = nextState;
	}
}

void printState(EightPuzzleState* state)
{

	for (int i = 0; i < 3; ++i)
	{
		cout << state->state[i][0] << " " << state->state[i][1] << " " << state->state[i][2] << endl;
	}
	cout << "---------------" << endl;
}