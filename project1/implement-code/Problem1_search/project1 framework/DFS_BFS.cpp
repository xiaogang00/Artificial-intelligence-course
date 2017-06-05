#include "DFS_BFS.h"
#include <iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<queue>
#include<stack>

using namespace std;

bool vis[maxn];  //for each node in the graph, mark whether the node has been visited during searching.
char node[6] = { 'A', 'B', 'C', 'D', 'E', 'F' };  //node symbol in order, used for visualization of searching result.


void createMGraph(MGraph &G) 
{
	cout << "enter the number of nodes in the graph:" << endl;
	cin >> G.n;
	cout << "enter the number of edges in the graph:" << endl;
	cin >> G.e;

	int i, j;
	int s, t;  //start and end node number
	int v;  //value of edge between node s and t
	for (i = 0; i<G.n; i++)  
	{
		vis[i] = false;
		for (j = 0; j<G.n; j++)
		{
			G.edges[i][j] = 0;
		}
	}
	cout << "enter the value of edge/weights and corresponding node number" << endl;
	cout << "(e.g. 1 3 10 means the edge from node 1 to node 3 is 10) : " << endl;
	cout << "first: " << endl;
	for (i = 0; i<G.e; i++) //set value for non-zero edges
	{
		cin >> s >> t >> v;    
		G.edges[s][t] = v;

		cout << "next:" << endl;
	}
}
/* Your implmentation of DFS, BFS for searching 
*/

/* Recursive implementation of DFS for finding path from given start node to end node
G: the graph for searching
start: start node number for searching
end: end node number for searching
endflag: a bool variable for determining whether searching has reached the end node
*/
void dfs(MGraph G, int start, int end, bool &endflag)
{
	cout << start <<endl;
	vis[start] = true;
	printf("%d  ", start);
	for (int i = 0 ; i < G.n ; i++)
	{
        if (G.edges[start][i] != 0 )
		{
			if (!vis[i])
			{
				vis[i] = true;
				printf("%d  ", i);
				dfs(G, i, end, endflag);
				if (i == end)
				{
					endflag = true;
					return ;
				}

			}
		}
	}
	/* if endflag is true: return;

	print (start node);
	set vis[start] to true;

	for (the neiboring nodes of start node):
	  if not visited:
	      dfs(G, neighbor,end,endflag);
		  if neighbor=end node:
		     set endflag true;
			 return;
	*/
}

/* Non-recursive implementation of DFS for finding path from given start point to end point
   data structure: Stack (LIFO)
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
*/
void dfs1(MGraph G, int start, int end)  
{
	stack<int> s;

	vis[start] = true;
	if (start == end)
		return;
	s.push(start);
    //printf("%d   ", start);

	int temp;
	while (!s.empty())
	{
         temp = s.top();
		 int i;
		 for (i = 0 ; i < G.n ; i++)
		 {
			if (G.edges[temp][i] != 0 )
			{
				if (!vis[i])
				{
					vis[i] = true;
					//printf("%d   ", i);
					s.push(i);
					if (i == end)
						return;
					break;
				}
			}
		 }
		 if (i == G.n )
		   s.pop();

	}

	/* 
	check if start node is end node;

	s.push(start);  //push start node into stack

	while (!s.empty())
	
		get top node in the stack;

		for (all the neighboring nodes) //�����붥��i���ڵĶ���
		
			if (not visited)
			
				operations to the node	
	*/
}

/* BFS implementation for finding path from given start point to end point
   data structure: Queue (FIFO)
   G: the graph for searching
   start: start node number for searching
   end: end node number for searching
*/
void bfs(MGraph G, int start, int end)
{
	queue<int> Q;
	if (start == end)
		return;
	vis[start] = true;
	Q.push(start);
	//printf("%d   ", start);
	int temp;
	while (! Q.empty())
	{
		temp = Q.front();
		Q.pop();
		for (int i = 0 ; i < G.n ; i++)
		{
			if (G.edges[temp][i] != 0 )
			{
				if (!vis[i])
				{
					vis[i] = true;
					//printf("%d   ", i);
					if (i == end)
						return;
					Q.push(i);
				}
			}
		}

	}


	/*
	set vis[start] to true;
	Q.push(start);  push start node to stack

	while (!Q.empty())
	
		get top node;

		for (all the neighbor node of top node) 
		  //BFS traversal
	
	*/
}

