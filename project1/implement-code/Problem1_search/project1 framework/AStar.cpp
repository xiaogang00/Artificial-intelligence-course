#include "AStar.h"
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

char AStarNode[6] = { 'A', 'B', 'C', 'D', 'E', 'F' };


void createGraph(Graph &G)
{
	cout << "enter the number of nodes in the graph:" << endl;
	cin >> G.n;
	cout << "enter the number of edges in the graph:" << endl;
	cin >> G.e;

	int i, j;
	int s, t; //start and end node number
	int v;  //value of edge between node s and t
	for (i = 0; i<G.n; i++)  
	{
		G.H[i] = 0;
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

	int number;
	cout << "Please give heuristic function value from all nodes to the goal node" << endl;
	for (i = 0; i < G.n; i++)
	{
		cin >> number;
		G.H[i] = number;
	}
}


void reconstruct_path(GraphNode* current, GraphNode & initNode, vector<int> &resultPath)
{
	resultPath.push_back(current->nodeNum);
	GraphNode* temp = current->preGraphNode;

	while(!(*temp == initNode))
	{
		resultPath.push_back(temp->nodeNum);
		temp = temp->preGraphNode;
	}
	resultPath.push_back(temp->nodeNum);
	//resultPath.reserve(resultPath.size());
	reverse(resultPath.begin(), resultPath.end());
}


void getSuccessors(GraphNode* current, Graph& graph, vector<GraphNode*> &successors)
{
	successors.clear();
    int i;
    for (i = 0; i < graph.n; i++)
    {
		if (graph.edges[current->nodeNum][i] != 0)
		{
			GraphNode* temp = new GraphNode;
			temp->nodeNum = i;
			successors.push_back(temp);
		}
    }
}


bool find_openlist(priority_queue<GraphNode*, vector<GraphNode*>, cmpLarge>& list, GraphNode* Node)
{
    vector<GraphNode*> member;
    bool flag = false;
    while(! list.empty())
    {
        GraphNode * temp = list.top();
		list.pop();
        member.push_back(temp);
        if(temp ->nodeNum == Node->nodeNum)
        {
            flag = true;
			int number = member.size();
			for (int i = 0; i < number; i++)
            {
                list.push(member[i]);
            }
            break;
        }
    }

	if (flag == false)
	{
		int number = member.size();
		for (int i = 0; i < number; i++)
		{
			list.push(member[i]);
		}
	}

    return flag;
}

bool find_closelist(vector<GraphNode*>& closelist, GraphNode* Node)
{
    bool flag = false;
	int number = closelist.size();

	for (int i = 0; i < number; i++)
	{
        GraphNode * temp = closelist[i];

        if(temp ->nodeNum == Node->nodeNum)
        {
            flag = true;
			break;
        }
    }
    return flag;
}

void AStarShortestPathSearch( Graph& graph, GraphNode& initNode, GraphNode& finalNode, vector<int> &resultPath){

	//push states to priority queue,and pop according to priority
	priority_queue<GraphNode*, vector<GraphNode*>, cmpLarge> openlist; 
	vector<GraphNode*> closelist;
	initNode.h  = graph.H[initNode.nodeNum];
	finalNode.h = graph.H[finalNode.nodeNum];

	openlist.push(&initNode); //取地址操作
	while (! openlist.empty())
	{
		GraphNode* current = openlist.top();
		openlist.pop();

		if (checkGoalNode(current, &finalNode))
		{
			return reconstruct_path(current, initNode, resultPath);
		}
		closelist.push_back(current);

        vector<GraphNode*> successors;
		getSuccessors(current, graph, successors);
        //GraphNode * succ = new GraphNode[successors.size()];

		int i;
		for (i = 0 ;i < successors.size() ; ++i)
		{
            bool result1 = find_closelist(closelist,successors[i]);
			bool result2 = find_openlist(openlist,successors[i]);
			if(result1)
			{
				continue;
			}
			if(!result2)
			{
				successors[i]->g = current->g + graph.edges[current->nodeNum][successors[i]->nodeNum];
				printf("%d   ", successors[i]->g);
				successors[i]->h = graph.H[successors[i]->nodeNum];
				printf("%d   ", successors[i]->h);
				successors[i]->preGraphNode = current;
				openlist.push(successors[i]);
			}
            else if ((current->g + graph.edges[current->nodeNum][successors[i]->nodeNum])>= successors[i]->g)
                continue;

			successors[i]->g = current->g + graph.edges[current->nodeNum][successors[i]->nodeNum];
			successors[i]->h = graph.H[successors[i]->nodeNum];
			successors[i]->preGraphNode = current;
		}
	}

	/*
	openlist.push(&initNode);
	while openlist is not empty:

		current = openlist.top()
		openlist.pop();

		if (current==goal node)
			return reconstruct_path(current, initNode, resultPath)

		closelist.push_back(current)

		successors=getSuccessors(current)
			for (i = 0; i < successors.size(); ++i)

				if successors[i] in closelist //if the successor has already been explored
					continue;
				if successors[i] not in openlist //check if the successor is not in the openlist
					compute g, h and preGraphNode of successors[i]
					push successors[i] to openlist
				
				else if g of successors[i] from current path is bigger than previous g of successors[i]
				//remember to set proper value to g and h 					continue;
				update g, h and preGraphNode of successors[i]
	*/		
}

bool checkGoalNode(const GraphNode* resultState, const GraphNode* goalNode){
	if (resultState->nodeNum == goalNode->nodeNum)
		return true;

	return false;
}

/* display the search result
   resultPath: a vector which stores the node numbers in order along the shortest path found
   MGraph : the search graph, here it is used to compute the path length
*/
bool displaySearchPath(const vector<int> &resultPath,const  Graph& MGraph){
	int shortestPath = 0;
	int num = int(resultPath.size());
	if (resultPath.empty())
		return false;
	
	cout << "The shortest path is:" << endl;
	for (int i = 0; i < num - 1; ++i){
		cout << AStarNode[resultPath[i]] << "-> ";
		shortestPath += MGraph.edges[resultPath[i]][resultPath[i + 1]];

	}
	cout << AStarNode[resultPath[num - 1]] << endl;

	cout << "Path length: " << shortestPath << endl;

	return true;
}



