#include "Astar.h"
#include "DFS_BFS.h"
#include <iostream>


int main()
{

	//MGraph graph;
	Graph graph;
	//createMGraph(graph);
	createGraph(graph);
	int start, end;

	cout << "Please choose initial node number:" << endl;
	cin >> start;

	cout << "Please choose goal node number:" << endl;
	cin >> end;

	//bool endflag = false;
	//dfs(graph, start, end,endflag);
	//dfs1(graph, start, end);
	//bfs(graph, start, end);
	vector<int> resultPath;
	GraphNode initNode, finalNode;
	initNode.nodeNum = start;
	finalNode.nodeNum = end;


	AStarShortestPathSearch(graph, initNode, finalNode, resultPath);
	bool success = displaySearchPath(resultPath, graph);
	if (success = true)
		cout << "Search succeeds" << endl;


	getchar(); 
	system("pause");
	return 0;
}