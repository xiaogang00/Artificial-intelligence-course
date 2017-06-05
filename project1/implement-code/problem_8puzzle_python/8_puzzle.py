import Queue

class Board(object):
    N = 3
    #matrix = [[0 for i in range(N)] for j in range(N)]
    #posx = 0
    #posy = 0
    def __init__(self, block):
        self.matrix = [[ 0 for i in range(self.N)] for j in range(self.N)]
        self.posx = 0
        self.posy = 0
        for i in range(self.N):
            for j in range(self.N):
                self.matrix[i][j] = block[i][j]
                if self.matrix[i][j] == 0:
                    self.posx = i
                    self.posy = j
        self.neigh = [[[ 0 for i in range(self.N)] for j in range(self.N)] for i in range(4)]

    def hamming(self):
        hammingDis = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.matrix[i][j] == 0:
                    continue
                if i * self.N +j +1 != self.matrix[i][j]:
                    hammingDis += 1
        return hammingDis

    def manhattan(self):
        manhattandis = 0
        count = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.matrix[i][j] == 0:
                    count+=1
                    continue
                if self.matrix[i][j] % self.N == 0:
                    x = self.matrix[i][j] / self.N - 1
                    y = self.N - 1
                else:
                    x = self.matrix[i][j] / self.N
                    y = self.matrix[i][j] % self.N - 1
                manhattandis += abs(i - x) + abs(j - y)
        if count == self.N * self.N :
            return count * count
        return manhattandis


    def isGoal(self):
        if self.posx != (self.N - 1) or (self.posy != self.N - 1):
            return 0
        count = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.matrix[i][j] == 0:
                    count += 1
                    continue
                if i * self.N + j + 1 != self.matrix[i][j]:
                    return 0

        if count == self.N * self.N:
            return 0

        return 1

    def exchange(self):
        x = -1
        y = -1
        temp_block = [[0 for i in range(self.N)] for j in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if j < (self.N - 1) and self.matrix[i][j] != 0 and self.matrix[i][j+1] != 0:
                    x = i
                    y = j
                temp_block[i][j] = self.matrix[i][j]

        t = temp_block[x][y]
        temp_block[x][y] = temp_block[x][y+1]
        temp_block[x][y+1] = t
        return temp_block

    def equals(self, y):
        if y == self:
            return 1
        if self.N != y.N:
            return 0
        for i in range(self.N):
            for j in range(self.N):
                if self.matrix[i][j] != y.matrix[i][j]:
                    return 0
        return 1


    def neighbors(self):
        dx = [0, 0, -1, 1]
        dy = [1, -1, 0, 0]
        for i in range(4):
            x = self.posx + dx[i]
            y = self.posy + dy[i]
            if x < self.N and x >= 0 and y < self.N and y >= 0:
                self.matrix[self.posx][self.posy], self.matrix[x][y] = self.matrix[x][y], self.matrix[self.posx][
                    self.posy]

                for j in range(self.N):
                    for k in range(self.N):
                        self.neigh[i][j][k] = self.matrix[j][k]

                self.matrix[self.posx][self.posy], self.matrix[x][y] = self.matrix[x][y], self.matrix[self.posx][
                    self.posy]

    def print_board(self):
        for i in range(self.N):
            for j in range(self.N):
                print self.matrix[i][j], '  ',
            print '\n'
        print '------------------------'


class Node(object):
    N = 3

    def __init__(self, puzzle):
        self.item = Board(puzzle)
        self.move = 0
        self.isTwin = 0
        #self.Prev = [[ 0 for i in range(self.N)] for j in range(self.N)]
        self.Prev = self

    def __cmp__(self, that):
        return cmp(self.move + self.item.manhattan(), that.move + that.item.manhattan())


class Solver(object):
    N = 3

    def __init__(self, initial):
        self.init = initial
        self.targetBoardNode = initial
        self.q = Queue.PriorityQueue()
        self.null_state = Node([[ 0 for i in range(self.N)] for j in range(self.N)])

    def solve(self, initial):
        while not self.q.empty():
             self.q.get()
        self.q.put(initial)
        twinbn = Node(initial.item.exchange())
        twinbn.isTwin = 1
        self.q.put(twinbn)

        while not self.q.empty():
            curbn = self.q.get()
            if curbn.isTwin == 0:
                print curbn.item.matrix
            if curbn.item.isGoal():
                if curbn.isTwin:
                    self.targetBoardNode.item.matrix == self.null_state
                else:
                    self.targetBoardNode = curbn
                break
            curbn.item.neighbors()
            for i in range(4):
                pre = curbn.Prev
                board = Board(curbn.item.neigh[i])
                if (curbn.Prev == self.null_state) or (not pre.item.equals(board)):
                    bn = Node(curbn.item.neigh[i])
                    bn.Prev = curbn
                    bn.move = curbn.move + 1
                    if curbn.isTwin == 1:
                        bn.isTwin = 1
                    else:
                        bn.isTwin = 0
                    self.q.put(bn)

    def isSolvable(self):
        if self.targetBoardNode.item.matrix != self.null_state:
            return 0
        return 1

    def solution(self):
        s = []
        count = 0
        tmpbn = self.targetBoardNode
        while (tmpbn != self.init):
            tmpbn.item.print_board()
            Prev = tmpbn.Prev
            tmpbn = Prev
            count += 1
        return count


if __name__ == '__main__':
    block = [[7, 1, 3], [4, 0, 2], [6, 8, 5]]
    initial = Node(block)
    solver = Solver(initial)
    solver.solve(initial)
    print solver.solution()
