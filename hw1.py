from maze import Maze
############# Write Your Library Here ##########
import math
################################################


def search(maze, func):
    return {
        "bfs": bfs,
        "ids":ids,
        "astar": astar,
        "astar_four_circles": astar_four_circles,
        "astar_many_circles": astar_many_circles
    }.get(func)(maze)

def bfs(maze:Maze):
    """
    [Problem 01] 제시된 stage1 맵 세 가지를 BFS를 구현하여 목표지점을 찾는 경로를 return하시오.
    """
    start_point=maze.startPoint()
    path=[]
    ####################### Write Your Code Here ################################

    #queue : 현재 탐색 중인 지점들을 저장하는 큐
    queue=[]

    #초기화 이후 시작 지점을 큐에 넣는다.
    queue.append(start_point)

    #visited : 이미 방문한 지점들을 저장하는 리스트
    visited=[]

    #시작 지점을 첫 방문 지점으로 추가한다.
    visited.append(start_point)

    #parent : 각 지점에 도달하기 직전에 탐색한 지점을 저장하는 2차원 배열
    parent=[]

    #parent를 초기화한다. 각 원소의 기본값은 자기 자신이다.
    for i in range(maze.rows):
        parent.append([])
        for j in range(maze.cols):
            parent[i].append((i, j))
    
    #BFS를 수행한다.
    while queue:
        #큐의 맨 앞에 있는 지점을 꺼낸다.
        current = queue.pop(0)

        #만약 목표 지점에 도달했다면 탐색을 종료한다.
        if maze.isObjective(current[0], current[1]):
            break

        #목표 지점이 아닐 경우, 현재 지점의 이웃 지점들을 탐색한다.
        for neighbor in maze.neighborPoints(current[0],current[1]):

            #아직 방문하지 않은 이웃 지점은 큐에 넣고, 방문한 지점으로 추가한다.
            if neighbor not in visited:
                queue.append(neighbor)
                visited.append(neighbor)

                #이웃 지점의 parent를 현재 지점으로 설정한다.
                parent[neighbor[0]][neighbor[1]] = current
    
    #목표 지점에서부터 역순으로 출발 지점까지의 경로를 구한다.
    path.append(current)

    #목표 지점에서 시작하여 직전에 탐색한 정점을 추적하여 출발 지점까지의 경로를 구한다. 
    #이하 나머지 알고리즘도 모두 동일한 방법을 사용한다.
    while parent[current[0]][current[1]] != current:
        current = parent[current[0]][current[1]]
        path.append(current)

    #path를 뒤집어 출발 지점부터 목표 지점까지의 경로를 구한다.
    path.reverse()

    #경로를 반환한다.
    return path
    ############################################################################

def ids(maze:Maze):
    """
    [Problem 02] 제시된 stage1 맵 세 가지를 IDS를 구현하여 목표지점을 찾는 경로를 return하시오.
    """
    start_point=maze.startPoint()
    path=[]
    ###################### Write Your Code Here ################################
    #dfs를 수행할 최대 깊이
    depth=0

    #depth를 1씩 증가시키면서 dfs를 수행한다.
    while True:
        #방문 여부를 확인하기 위한 배열
        visited=[]
        
        #각 점에 도달하기 직전에 있었던 지점을 저장하기 위한 2차원 배열.
        parent=[]

        #parent를 초기화한다. 각 원소의 기본값은 자기 자신이다.
        for i in range(maze.rows):
            parent.append([])
            for j in range(maze.cols):
                parent[i].append((i, j))
        
        #depth만큼 dfs를 수행한다. 만약 목표 지점에 도달했다면 break까지 수행하고 반복문을 탈출한다.
        if dfs(maze, visited, start_point, depth, parent):

            #답을 찾았다면, 마지막으로 방문한 노드는 목적지다.
            node = visited[-1]

            #목적지에서부터 역순으로 출발 지점까지의 경로를 구한다.
            path.append(node)
            while parent[node[0]][node[1]] != node:
                node = parent[node[0]][node[1]]
                path.append(node)

            #path를 뒤집어 출발 지점부터 목표 지점까지의 경로를 구한다.
            path.reverse()

            #경로를 구했으므로 탐색을 중단한다.
            break

        #현재 depth까지 탐색했을 때 답을 찾지 못했다면 depth를 1 증가시킨다.
        depth += 1

    #경로를 반환한다.
    return path

#dfs를 수행하는 함수. maze : 미로, visited : 방문 여부를 저장하는 리스트, current : 현재 지점, depth : 현재 depth, parent : 각 지점에 도달하기 직전에 있었던 지점을 저장하는 2차원 배열
def dfs(maze, visited, current, depth, parent):

    #우선 현재 지점을 방문한 것으로 처리한다.
    visited.append(current)

    #만약 목표 지점에 도달했다면 True를 반환한다.
    if maze.isObjective(current[0], current[1]):
        return True
    
    #현재 depth가 0이라면 더 이상 깊이 들어가지 않고 탐색을 중단한다.
    if depth <= 0:
        #방문한 지점에서 현재 지점을 제거한다.
        visited.pop()
        #목적지는 아니기 때문에 False를 반환한다.
        return False
    
    #현재 지점과 맞닿아 있는 이웃 지점들을 탐색한다.
    for neighbor in maze.neighborPoints(current[0], current[1]):

        #이웃 지점이 아직 방문하지 않은 지점이라면, 현재 지점을 이웃 지점의 parent로 설정하고 dfs를 수행한다.
        if neighbor not in visited:
            parent[neighbor[0]][neighbor[1]] = current

            #만약 목표 지점에 도달했다면 True를 반환한다.
            if dfs(maze, visited, neighbor, depth-1, parent):
                return True
            
    #cycle이 발생하면 depth는 0보다 큰데 더 이상 탐색할 곳이 없다. 따라서 방문한 지점에서 현재 지점을 제거하고 False를 반환한다.
    visited.pop()
    return False
    #############################################################################

# Manhattan distance
def stage1_heuristic(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def astar(maze:Maze):
    """
    [Problem 03] 제시된 stage1 맵 세가지를 A* Algorithm을 통해 최단경로를 return하시오.
    (Heuristic Function은 위에서 정의한 stage1_heuristic function(manhattan_dist)을 사용할 것.)
    """
    start_point = maze.startPoint()
    path = []
    ####################### Write Your Code Here ################################
    #queue : 현재 탐색 중인 지점들을 저장하는 큐
    queue = []

    #초기화 이후 시작 지점을 큐에 넣는다.
    queue.append(start_point)

    #visited : 이미 방문한 지점들을 저장하는 리스트
    visited=[]
    
    #시작 지점을 첫 방문 지점으로 추가한다.
    visited.append(start_point)

    #parent : 각 지점에 도달하기 직전에 탐색한 지점을 저장하는 2차원 배열
    parent=[]

    #parent를 초기화한다. 각 원소의 기본값은 자기 자신이다.
    for i in range(maze.rows):
        parent.append([])
        for j in range(maze.cols):
            parent[i].append((i, j))

    #previous_cost : 각 지점에 도달하기 직전까지의 cost를 저장하는 2차원 배열
    previous_cost=[]

    #previous_cost를 초기화한다. 각 원소의 기본값은 0이다.
    for i in range(maze.rows):
        previous_cost.append([])
        for j in range(maze.cols):
            previous_cost[i].append(0)

    #A*를 수행한다. 전체적인 수행 방식은 BFS와 비슷하다.
    while queue:

        #큐의 맨 앞에 있는 지점을 꺼낸다.
        current = queue.pop(0)

        #만약 해당 지점이 목적지라면, 탐색을 종료한다.
        if maze.isObjective(current[0], current[1]):
            break

        #목적지가 아니라면, 현재 지점의 이웃 지점들을 탐색한다.
        for neighbor in maze.neighborPoints(current[0],current[1]):

            #아직 방문하지 않은 이웃 지점은 큐에 넣고, 방문한 지점으로 추가한다.
            if neighbor not in visited:
                queue.append(neighbor)
                visited.append(neighbor)

                #이웃 지점의 parent를 현재 지점으로 설정한다.
                parent[neighbor[0]][neighbor[1]] = current

                #이웃 지점의 탐색 비용은 (현재 지점까지의 탐색 비용 + 1)이다.
                previous_cost[neighbor[0]][neighbor[1]] = previous_cost[current[0]][current[1]] + 1

        #A* 알고리즘에서 고려하는 비용은 f(n) = g(n) + h(n)이다. 여기서 g(n)은 출발 지점에서 현재 지점까지의 비용이고, h(n)은 현재 지점에서 목적지까지의 예상 비용이다.
        #cost는 f(n)을 저장하기 위한 Dictionary. cost[x]는 x 지점까지의 예상 비용을 의미한다.
        cost = {}

        #각 지점에 대해 f(n)을 계산한다. 이때 g(n)은 previous_cost를 통해 구하고, h(n)은 stage1_heuristic을 통해 구한다.
        for element in queue:
            cost[element] = previous_cost[element[0]][element[1]] + stage1_heuristic(element, maze.circlePoints()[0])

        #f(n)을 기준으로 큐를 정렬한다. 따라서 다음에 탐색할 지점은 queue 내에서 예상 비용이 가장 작은 지점이다.
        queue.sort(key=lambda x: cost[x])
    
    #목적지에서부터 역순으로 출발 지점까지의 경로를 구한다.
    node = visited[-1]
    path.append(node)
    while parent[node[0]][node[1]] != node:
        node = parent[node[0]][node[1]]
        path.append(node)
    
    #path를 뒤집어 출발 지점부터 목표 지점까지의 경로를 구한다.
    path.reverse()

    #경로를 반환한다.
    return path
    ############################################################################


####################### Write Your Code Here ################################
def stage2_heuristic(p1, p2):
    #euclidean distance
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
############################################################################


def astar_four_circles(maze:Maze):
    """
    [Problem 04] 제시된 stage2 맵 세 가지를 A* Algorithm을 통해 최단경로를 return하시오.
    (Heuristic Function은 직접 정의할것 )
    """
    start_point = maze.startPoint()
    path = []
    ####################### Write Your Code Here ###############################
    #circle_points: 미로에 있는 목적지들의 위치를 저장하는 리스트
    circle_points = maze.circlePoints()

    #출발 지점에서 시작하여 미로 내의 목적지들을 최단 경로로 모두 지나가는 경로를 구한다.
    #이를 위해 4개의 목적지 중에 출발 지점과 가장 가까운 목적지를 먼저 탐색하고, 그 다음으로 가까운 목적지를 탐색하는 방식을 사용한다.
    while circle_points:

        #현재 출발 지점에서 가장 가까운 목적지를 구한다.
        circle_points.sort(key=lambda x: stage2_heuristic(start_point, x))

        #목적지를 구했으므로, 해당 목적지를 출발 지점으로 설정한다.
        target = circle_points[0]

        #queue : 현재 탐색 중인 지점들을 저장하는 큐
        queue = []

        #초기화 이후 시작 지점을 큐에 넣는다.
        queue.append(start_point)

        #visited : 이미 방문한 지점들을 저장하는 리스트
        visited=[]

        #시작 지점을 첫 방문 지점으로 추가한다.
        visited.append(start_point)

        #parent : 각 지점에 도달하기 직전에 탐색한 지점을 저장하는 2차원 배열
        parent=[]

        #parent를 초기화한다. 각 원소의 기본값은 자기 자신이다.
        for i in range(maze.rows):
            parent.append([])
            for j in range(maze.cols):
                parent[i].append((i, j))

        #previous_cost : 각 지점에 도달하기 직전까지의 cost를 저장하는 2차원 배열
        previous_cost=[]

        #previous_cost를 초기화한다. 각 원소의 기본값은 0이다.
        for i in range(maze.rows):
            previous_cost.append([])
            for j in range(maze.cols):
                previous_cost[i].append(0)
        
        #A* 알고리즘을 수행한다.
        while queue:

            #큐의 맨 앞에 있는 지점을 꺼낸다.
            current = queue.pop(0)

            #만약 해당 지점이 목적지라면, 탐색을 종료한다.
            if current is target:
                break

            #목적지가 아니라면, 현재 지점의 이웃 지점들을 탐색한다.
            for neighbor in maze.neighborPoints(current[0],current[1]):

                #아직 방문하지 않은 이웃 지점은 큐에 넣고, 방문한 지점으로 추가한다.
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.append(neighbor)

                    #이웃 지점의 parent를 현재 지점으로 설정한다.
                    parent[neighbor[0]][neighbor[1]] = current

                    #이웃 지점의 탐색 비용은 (현재 지점까지의 탐색 비용 + 1)이다.
                    previous_cost[neighbor[0]][neighbor[1]] = previous_cost[current[0]][current[1]] + 1

            #A* 알고리즘에서 고려하는 비용은 f(n) = g(n) + h(n)이다. 여기서 g(n)은 출발 지점에서 현재 지점까지의 비용이고, h(n)은 현재 지점에서 목적지까지의 예상 비용이다.
            #cost는 f(n)을 저장하기 위한 Dictionary. cost[x]는 x 지점까지의 예상 비용을 의미한다.
            cost = {}

            #각 지점에 대해 f(n)을 계산한다. 이때 g(n)은 previous_cost를 통해 구하고, h(n)은 stage2_heuristic을 통해 구한다.
            for element in queue:
                cost[element] = previous_cost[element[0]][element[1]] + stage2_heuristic(element, target)

            #f(n)을 기준으로 큐를 정렬한다. 따라서 다음에 탐색할 지점은 queue 내에서 예상 비용이 가장 작은 지점이다.
            queue.sort(key=lambda x: cost[x])

        #현재의 목적지에서부터 현재의 출발 지점까지의 경로를 역순으로 구한다.
        part_path=[]
        node = target
        part_path.append(node)
        while parent[node[0]][node[1]] != node:
            node = parent[node[0]][node[1]]
            part_path.append(node)

        #part_path를 뒤집어 현재 출발 지점부터 목표 지점까지의 경로를 구한다.
        part_path.reverse()

        #전체 경로에 이 부분 경로를 더한다.
        path += part_path

        #현재의 목표 지점을 다음 출발 지점으로 설정한다. 이후 다음 목표 지점을 구할 때, 현재 출발 지점은 고려하지 않는다.
        start_point=circle_points.pop(0)

    #경로를 반환한다.
    return path
    ############################################################################


####################### Write Your Code Here ###############################
#MST를 이용하여 다음 경로를 고려하기 위한 heuristic function
def stage3_heuristic(maze, start, goal):
    #MST를 이용해서 최단 경로를 에측하는 방법은 Dijkstra's algorithm과 비슷하게, 미로(maze) 내에서 현재 지점(start)에서 목표 지점(goal)까지의 MST를 만들어보는 것이다.
    #이때 MST를 만들기 위해 사용하는 비용을 반환한다.

    #MST를 구할 때는 bfs를 사용한다.
    #queue : 현재 탐색 중인 지점들을 저장하는 큐
    queue=[]

    #초기화 이후 시작 지점을 큐에 넣는다.
    queue.append(start)

    #visited : 이미 방문한 지점들을 저장하는 리스트
    visited=[]

    #시작 지점을 첫 방문 지점으로 추가한다.
    visited.append(start)

    #mst_cost : 현재 지점에서 목표 지점까지의 MST를 만들기 위해 사용하는 비용을 저장하는 변수.
    #이 비용은 MST를 만들기 위해 수행한 BFS 내부의 연산의 횟수다.
    mst_cost=0

    #BFS를 수행한다.
    while queue:

        #큐의 맨 앞에 있는 지점을 꺼낸다. 하나의 연산으로 간주한다.
        mst_cost += 1
        current = queue.pop(0)

        #만약 목표 지점에 도달했다면 탐색을 종료한다. isObjective() 메소드 실행을 하나의 연산으로 간주한다.
        mst_cost += 1
        if maze.isObjective(current[0], current[1]):
            break
        
        #목표 지점이 아닐 경우, 현재 지점의 이웃 지점들을 탐색한다. neighborPoints() 메소드 실행을 하나의 연산으로 간주한다.
        mst_cost += 1
        for neighbor in maze.neighborPoints(current[0],current[1]):

            #아직 방문하지 않은 이웃 지점은 큐에 넣고, 방문한 지점으로 추가한다. neighbor 탐색, queue.append(), visited.append() 메소드 실행을 각각 하나의 연산으로 간주한다.
            mst_cost += 1
            if neighbor not in visited:
                queue.append(neighbor)
                visited.append(neighbor)
                mst_cost +=2

    #현재 지점에서 MST를 구한다.
    return mst_cost
############################################################################

def astar_many_circles(maze: Maze):
    """
    [Problem 05] 제시된 stage3 맵 다섯 가지를 A* Algorithm을 통해 최단 경로를 return하시오.
    (Heuristic Function은 직접 정의 하고, minimum spanning tree를 활용하도록 한다.)
    """
    start_point = maze.startPoint()
    path = []
    ####################### Write Your Code Here ################################
    #circle_points: 미로에 있는 목적지들의 위치를 저장하는 리스트
    circle_points = maze.circlePoints()

    #출발 지점에서 시작하여 미로 내의 목적지들을 최단 경로로 모두 지나가는 경로를 구한다.
    #이를 위해 목적지들 중에 출발 지점과 가장 가까운 목적지를 먼저 탐색하고, 그 다음으로 가까운 목적지를 탐색하는 방식을 사용한다.
    while circle_points:

        #현재 출발 지점에서 가장 가까운 목적지를 구한다.
        circle_points.sort(key=lambda x: stage3_heuristic(maze, start_point, x))

        #목적지를 구했으므로, 해당 목적지를 출발 지점으로 설정한다.
        target = circle_points[0]

        #queue : 현재 탐색 중인 지점들을 저장하는 큐
        queue = []

        #초기화 이후 시작 지점을 큐에 넣는다.
        queue.append(start_point)

        #visited : 이미 방문한 지점들을 저장하는 리스트
        visited=[]

        #시작 지점을 첫 방문 지점으로 추가한다.
        visited.append(start_point)

        #parent : 각 지점에 도달하기 직전에 탐색한 지점을 저장하는 2차원 배열
        parent=[]

        #parent를 초기화한다. 각 원소의 기본값은 자기 자신이다.
        for i in range(maze.rows):
            parent.append([])
            for j in range(maze.cols):
                parent[i].append((i, j))

        #previous_cost : 각 지점에 도달하기 직전까지의 cost를 저장하는 2차원 배열
        previous_cost=[]

        #previous_cost를 초기화한다. 각 원소의 기본값은 0이다.
        for i in range(maze.rows):
            previous_cost.append([])
            for j in range(maze.cols):
                previous_cost[i].append(0)
        
        #A* 알고리즘을 수행한다.
        while queue:

            #큐의 맨 앞에 있는 지점을 꺼낸다.
            current = queue.pop(0)

            #만약 해당 지점이 목적지라면, 탐색을 종료한다.
            if current is target:
                break

            #목적지가 아니라면, 현재 지점의 이웃 지점들을 탐색한다.
            for neighbor in maze.neighborPoints(current[0],current[1]):

                #아직 방문하지 않은 이웃 지점은 큐에 넣고, 방문한 지점으로 추가한다.
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.append(neighbor)

                    #이웃 지점의 parent를 현재 지점으로 설정한다.
                    parent[neighbor[0]][neighbor[1]] = current

                    #이웃 지점의 탐색 비용은 (현재 지점까지의 탐색 비용 + 1)이다.
                    previous_cost[neighbor[0]][neighbor[1]] = previous_cost[current[0]][current[1]] + 1

            #A* 알고리즘에서 고려하는 비용은 f(n) = g(n) + h(n)이다. 여기서 g(n)은 출발 지점에서 현재 지점까지의 비용이고, h(n)은 현재 지점에서 목적지까지의 예상 비용이다.
            #cost는 f(n)을 저장하기 위한 Dictionary. cost[x]는 x 지점까지의 예상 비용을 의미한다.
            cost = {}

            #각 지점에 대해 f(n)을 계산한다. 이때 g(n)은 previous_cost를 통해 구하고, h(n)은 stage3_heuristic을 통해 구한다.
            for element in queue:
                cost[element] = previous_cost[element[0]][element[1]] + stage3_heuristic(maze, element, target)

            #f(n)을 기준으로 큐를 정렬한다. 따라서 다음에 탐색할 지점은 queue 내에서 예상 비용이 가장 작은 지점이다.
            queue.sort(key=lambda x: cost[x])

        #현재의 목적지에서부터 현재의 출발 지점까지의 경로를 역순으로 구한다.
        part_path=[]
        node = target
        part_path.append(node)
        while parent[node[0]][node[1]] != node:
            node = parent[node[0]][node[1]]
            part_path.append(node)

        #part_path를 뒤집어 현재 출발 지점부터 목표 지점까지의 경로를 구한다.
        part_path.reverse()

        #전체 경로에 이 부분 경로를 더한다.
        path += part_path

        #현재의 목표 지점을 다음 출발 지점으로 설정한다. 이후 다음 목표 지점을 구할 때, 현재 출발 지점은 고려하지 않는다.
        start_point=circle_points.pop(0)

    #경로를 반환한다.
    return path
    ############################################################################
    
