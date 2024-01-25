import numpy as np

epsilon = 1e-3


def compute_transition_matrix(model):
    """
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    """

    # M과 N을 각각 변수로 사용하기 위해 model.R.shape을 사용한다.
    (M, N) = model.R.shape

    # P를 영행렬로 정의한다.
    P = np.zeros((M, N, 4, M, N))

    # left, up, right, down | 각 방향 별 좌표의 변화를 리스트에 저장한다.
    Dir = [(0, -1), (-1, 0), (0, 1), (1, 0)]

    for i in range(M):
        for j in range(N):

            # (r, c)가 Terminal State가 아닐 때 확률을 계산한다. Terminal State일 경우, P(r, c, x, x, x) = 0이다.
            if model.T[i][j] == False:

                # 모든 방향에 대햐여 이동 확률을 계산한다.
                for k in range(4):

                    # 1. 에이전트가 의도한 방향 그대로 이동하는 경우
                    # 1-1. 이동한 위치가 주어진 grid world 안에 있는 경우
                    if i+Dir[k][0] in range(M) and j+Dir[k][1] in range(N):

                        # 이동한 위치가 벽이 아닌 경우
                        if model.W[i+Dir[k][0]][j+Dir[k][1]] == False:
                            # P(r, c, a, r', c') = D(i, j, 0)
                            P[i][j][k][i+Dir[k][0]][j+Dir[k][1]] += model.D[i][j][0]
                        
                        # 이동한 위치가 벽인 경우
                        else:
                            # 현재 state를 유지한다.
                            P[i][j][k][i][j] += model.D[i][j][0]

                    # 1-2. 이동한 위치가 주어진 grid world 바깥인 경우
                    else:
                        # 현재 state를 유지한다.
                        P[i][j][k][i][j] += model.D[i][j][0]

                    # clockwise: 시계 방향, anticlockwise: 반시계 방향
                    clockwise = (k+1)%4
                    anticlockwise = k-1 if k-1 >= 0 else 3-k

                    # 2. 에이전트가 의도한 방향의 반시계 방향으로 이동하는 경우
                    # 2-1. 이동한 위치가 주어진 grid world 안에 있는 경우
                    if i+Dir[anticlockwise][0] in range(M) and j+Dir[anticlockwise][1] in range(N):

                        # 이동한 위치가 벽이 아닌 경우
                        if model.W[i+Dir[anticlockwise][0]][j+Dir[anticlockwise][1]] == False:
                            # P(r, c, a, r', c') = D(i, j, 1)
                            P[i][j][k][i+Dir[anticlockwise][0]][j+Dir[anticlockwise][1]] += model.D[i][j][1]

                        # 이동한 위치가 벽인 경우
                        else:
                            # 현재 state를 유지한다.
                            P[i][j][k][i][j] += model.D[i][j][1]
                    
                    # 2-2. 이동한 위치가 주어진 grid world 바깥인 경우
                    else:
                        # 현재 state를 유지한다.
                        P[i][j][k][i][j] += model.D[i][j][1]

                    # 3. 에이전트가 의동한 방향의 시계 방향으로 이동한 경우
                    # 3-1. 이동한 위치가 주어진 grid world 안에 있는 경우
                    if i+Dir[clockwise][0] in range(M) and j+Dir[clockwise][1] in range(N):

                        # 이동한 위치가 벽이 아닌 경우
                        if model.W[i+Dir[clockwise][0]][j+Dir[clockwise][1]] == False:
                            # P(r, c, a, r', c') = D(i, j, 2)
                            P[i][j][k][i+Dir[clockwise][0]][j+Dir[clockwise][1]] += model.D[i][j][2]
                        
                        # 이동한 위치가 벽인 경우
                        else:
                            # 현재 state를 유지한다.
                            P[i][j][k][i][j] += model.D[i][j][2]
                    
                    # 3-2. 이동한 위치가 주어진 grid world 바깥인 경우
                    else:
                        # 현재 state를 유지한다.
                        P[i][j][k][i][j] += model.D[i][j][2]
    
    # 완성된 P를 반환한다.
    return P

def update_utility(model, P, U_current):
    """
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    """

    # U_next(s) = R(s) + gamma * max(sum(P(s, a, s') * U_current(s'))), 모든 state를 포함하는 M X N 행렬을 한 번에 계산한다.
    # axis=(-1, -2): (M, N, 4, M, N) -> (M, N, 4) | s'에 대한 sum
    # axis=2: a를 기준으로 비교. (r, c, a, r', c') | axis = (0, 1, 2, 3, 4)
    U_next = model.R + model.gamma * np.max(np.sum(P * U_current, axis=(-1, -2)), axis=2)

    # 완성된 U_next를 반환한다.
    return U_next


def value_iteration(model):
    """
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    """

    # P를 미리 계산한다.
    P = compute_transition_matrix(model)

    # M과 N을 각각 변수로 사용하기 위해 model.R.shape을 사용한다.
    (M, N) = model.R.shape

    # U_current, U_next를 영행렬로 정의한다.
    U_current = np.zeros((M, N))
    U_next = np.zeros((M, N))

    # EPS는 |U_next(s) - U_current(s)|과 epsilon의 크기 비교를 추적한다.
    # |U_next(s) - U_current(s)| < epsilon일 때 True, 나머지는 False.

    # 에이전트가 가질 수 있는 모든 state에 대하여 최대 Utility를 구한다.
    EPS = False
    while EPS == False:

        # utility를 지속적으로 갱신하기 위해 update_utility 함수를 적용한 결과를 현재 상태로 재설정한다.
        U_current = U_next

        # 현재 상태에서 update_utility 함수를 적용한다.
        U_next = update_utility(model, P, U_current)

        EPS = True
        # 모든 State s에 대하여 |U_next(s) - U_current(s)| 과 epsilon을 비교한다.
        for i in range(M):
            for j in range(N):

                # 만약 모든 상태 중 하나라도 |U_next(s) - U_current(s)| < epsilon이 성립하지 않는다면, 최적 상태에 도달하지 않은 것이다.
                if abs(U_next[i][j] - U_current[i][j]) >= epsilon:
                    EPS = False

    return U_next
