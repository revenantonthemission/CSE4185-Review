from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

## Example Agent
class ReflexAgent(Agent):

  def Action(self, gameState):

    move_candidate = gameState.getLegalActions()
    ## Compute score of each action
    scores = [self.reflex_agent_evaluationFunc(gameState, action) for action in move_candidate]
    bestScore = max(scores)
    Index = [index for index in range(len(scores)) if scores[index] == bestScore]
    ## Pick randomly among the best
    get_index = random.choice(Index)

    return move_candidate[get_index]

  def reflex_agent_evaluationFunc(self, currentGameState, action):

    ## Next state of Pacman after action
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    ## Position of Pacman after action
    newPos = successorGameState.getPacmanPosition()
    ## Food of current state
    oldFood = currentGameState.getFood()
    ## Next state of ghosts after action
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ## Score after action
    return successorGameState.getScore()



def scoreEvalFunc(currentGameState):

  return currentGameState.getScore()

class AdversialSearchAgent(Agent):

  def __init__(self, getFunc ='scoreEvalFunc', depth ='2'):
    self.index = 0
    self.evaluationFunction = util.lookup(getFunc, globals())

    self.depth = int(depth)



class MinimaxAgent(AdversialSearchAgent):
  """
    [문제 01] MiniMaxAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    ## MAX가 가장 유리한 선택지를 계산한다.
    # self: 현재의 MinimaxAgent
    # currentGameState: 현재의 게임 상태
    # depth: 현재의 탐색 깊이
    def max_agent_evaluationFunc(self, currentGameState, depth):

      ## 정해진 깊이에 도달하거나 MAX가 게임 오버 상황에 처한 경우 앞으로의 탐색은 하지 않는다.
      if depth == 0 or currentGameState.isWin() or currentGameState.isLose():

        ## 이때 계산한 점수와 움직임을 반환한다.
        return (self.evaluationFunction(currentGameState), None)
      
      ## value: MAX가 가장 유리한 선택을 했을 때의 점수.
      value = float("-inf")

      ## move: MAX의 가장 유리한 선택.
      move = Directions.STOP

      #현재 상태에서 MAX가 내릴 수 있는 모든 선택에 대하여 그 이후의 상태를 계산한다.
      for action in currentGameState.getLegalActions():

        ## successorGameState: 현재 선택을 실행한 이후의 상태.
        successorGameState = currentGameState.generateSuccessor(0, action)

        ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 그때 MIN의 움직임은 tmp_move.
        (tmp, tmp_move) = min_agent_evaluationFunc(self, successorGameState, depth, 1)

        ## 만약 tmp가 기존의 최고값보다 크다면, 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
        if tmp > value:
          value = tmp
          move = action
      
      ## MAX에게 가장 유리한 선택을 반환한다.
      return (value, move)
  
    ## MIN에게 가장 유리한 선택지를 계산한다.
    # self: 현재의 MinimaxAgent
    # currentGameState: 현재의 게임 상태
    # depth: 현재의 탐색 깊이
    # agentIndex: 현재의 에이전트 인덱스
    def min_agent_evaluationFunc(self, currentGameState, depth, agentIndex):
      
      ## 정해진 깊이에 도달하거나 게임 오버 상황에 처한 경우 앞으로의 탐색은 하지 않는다.
      if depth == 0 or currentGameState.isWin() or currentGameState.isLose():

        ## 이때 계산한 점수와 움직임을 반환한다.
        return (self.evaluationFunction(currentGameState), None)
      
      ## value: MIN이 자신에게 자장 유리한 선택을 했을 때의 점수. MIN에게 유리할 수록 MAX에게 불리하기 때문에 최소값을 만들어내는 선택지를 찾아야 한다.
      value = float("inf")

      ## move: MIN에게 가장 유리한 선택지.
      move = Directions.STOP

      ## 만약 마지막 MIN (팩맨 게임에서는 유령)을 고려하고 있다면, 다음에는 깊이를 한 단계 더 늘리고 MAX의 다음 움직임을 고려해 마지막 MIN의 선택을 결정한다.
      if agentIndex == currentGameState.getNumAgents()-1:

        ## 현재의 MIN이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
        for action in currentGameState.getLegalActions(agentIndex):

          ## successorGameState: 현재 선택을 실행한 이후의 상태.
          successorGameState = currentGameState.generateSuccessor(agentIndex, action)

          ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 점수는 tmp, MAX의 움직임은 tmp_move.
          (tmp, tmp_move) = max_agent_evaluationFunc(self, successorGameState, depth-1)

          ## 만약 tmp가 기존의 최소값보다 작다면, MIN의 입장에서 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
          if tmp < value:
            value = tmp
            move = action
      
      ## 아직 모든 MIN(팩맨 게임의 유령)을 고려하지 않았다면 depth를 유지하고 다음 MIN의 선택을 고려한다.
      else:

        ## 현재의 MIN이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
        for action in currentGameState.getLegalActions(agentIndex):

          ## successorGameState: 현재 선택을 실행한 이후의 상태.
          successorGameState = currentGameState.generateSuccessor(agentIndex, action)

          ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 최선의 움직임은 tmp_move.
          (tmp, tmp_move) = min_agent_evaluationFunc(self, successorGameState, depth, agentIndex+1)

          ## 만약 tmp가 기존의 최소값보다 작다면, MIN의 입장에서 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
          if tmp < value:
            value = tmp
            move = action

      ## MIN에게 가장 유리한 선택을 반환한다.
      return (value, move)
    
    ## Minimax 알고리즘을 통해 MAX가 가장 유리한 선택지를 계산한다.
    (score, move) = max_agent_evaluationFunc(self, gameState, self.depth)

    ## MAX의 다음 움직임을 반환한다.
    return move
    ############################################################################




class AlphaBetaAgent(AdversialSearchAgent):
  """
    [문제 02] AlphaBetaAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    ## MAX가 가장 유리한 선택지를 계산한다.
    # self: 현재의 AlphaBetaAgent
    # currentGameState: 현재의 게임 상태
    # depth: 현재의 탐색 깊이
    # alpha: 탐색 과정에서 여태까지 MAX가 내린 선택의 결과 중 최선을 alpha라 부른다.
    # beta: 탐색 과정에서 여태까지 MIN이 내린 선택의 결과 중 최선을 beta라 부른다.
    def max_agent_evaluationFunc(self, currentGameState, depth, alpha, beta):

      ## 정해진 깊이에 도달하거나 게임 오버 상황에 처한 경우 앞으로의 탐색은 하지 않는다.
      if depth == 0 or currentGameState.isWin() or currentGameState.isLose():

        ## 이때 계산한 점수와 움직임을 반환한다.
        return (self.evaluationFunction(currentGameState), None)
      
      ## value: MAX이 자신에게 자장 유리한 선택을 했을 때의 점수.
      value = float("-inf")
      
      ## move: MAX에게 가장 유리한 선택지.
      move = Directions.STOP

      ## 현재의 MAX이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
      for action in currentGameState.getLegalActions():

        ## successorGameState: 현재 선택을 실행한 이후의 상태.
        successorGameState = currentGameState.generateSuccessor(0, action)

        ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 최선의 움직임은 tmp_move.
        (tmp, tmp_move) = min_agent_evaluationFunc(self, successorGameState, depth, 1, alpha, beta)

        ## 만약 tmp가 기존의 최고값보다 크다면, 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
        if tmp > value:
          value = tmp
          move = action
          ## 현재 선택과 기존의 선택을 비교하여 더 높은 점수를 내는 선택지의 점수를 alpha로 한다.
          alpha = max(alpha, value)
        
        ## MAX의 현재 선택 이후 MIN이 내린 최선의 선택이 MAX가 내린 다른 선택 이후 MIN이 내린 최선의 선택보다 못하다면, 이 선택지는 고려할 필요가 없기 때문에 즉시 반환한다.
        if value >= beta:
          return (value, move)

      ## MAX의 다음 움직임을 반환한다.
      return (value, move)
  
    ## MIN에게 가장 유리한 선택지를 계산한다.
    def min_agent_evaluationFunc(self, currentGameState, depth, agentIndex, alpha, beta):
      
      ## 정해진 깊이에 도달하거나 게임 오버 상황에 처한 경우 앞으로의 탐색은 하지 않는다.
      if depth == 0 or currentGameState.isWin() or currentGameState.isLose():

        ## 이때 계산한 점수와 움직임을 반환한다.
        return (self.evaluationFunction(currentGameState), None)
      
      ## value: MIN이 자신에게 자장 유리한 선택을 했을 때의 점수. MIN에게 유리할 수록 MAX에게 불리하기 때문에 최소값을 만들어내는 선택지를 찾아야 한다.
      value = float("inf")

      ## move: MIN에게 가장 유리한 선택지.
      move = Directions.STOP

      ## 만약 마지막 MIN (팩맨 게임에서는 유령)을 고려하고 있다면, 다음에는 깊이를 한 단계 더 늘리고 MAX의 다음 움직임을 고려해 마지막 MIN의 선택을 결정한다.
      if agentIndex == currentGameState.getNumAgents()-1:

        ## 현재의 MIN이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
        for action in currentGameState.getLegalActions(agentIndex):

          ## successorGameState: 현재 선택을 실행한 이후의 상태.
          successorGameState = currentGameState.generateSuccessor(agentIndex, action)

          ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 최선의 움직임은 tmp_move.          
          (tmp, tmp_move) = max_agent_evaluationFunc(self, successorGameState, depth-1, alpha, beta)

          ## 만약 tmp가 기존의 최소값보다 작다면, MIN의 입장에서 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
          if tmp < value:
            value = tmp
            move = action
            ## beta는 현재 선택 이후 MIN이 내릴 수 있는 최선의 선택을 의미한다. 기존의 최저 점수와 현재의 최저 점수를 비교하여 더 작은 쪽을 beta로 결정한다.
            beta = min(beta, value)

          ## 만약 MIN의 현재 선택 이후 MAX가 내린 최고의 선택이 MIN의 다른 선택에서 MAX가 내린 최고의 선택만 못하다면, 이후의 선택은 고려할 필요가 없기 때문에 바로 반환한다.
          if value < alpha:
            return (value, move)
      
      ## 아직 모든 MIN(팩맨 게임의 유령)을 고려하지 않았다면 depth를 유지하고 다음 MIN의 선택을 고려한다. 
      else:

        ## 현재의 MIN이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
        for action in currentGameState.getLegalActions(agentIndex):

          ## successorGameState: 현재 선택을 실행한 이후의 상태.
          successorGameState = currentGameState.generateSuccessor(agentIndex, action)

          ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 최선의 움직임은 tmp_move.
          (tmp, tmp_move) = min_agent_evaluationFunc(self, successorGameState, depth, agentIndex+1, alpha, beta)

          ## 만약 tmp가 기존의 최소값보다 작다면, MIN의 입장에서 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
          if tmp < value:
            value = tmp
            move = action
            ## beta는 현재 선택 이후 MIN이 내릴 수 있는 최선의 선택을 의미한다. 기존의 최저 점수와 현재의 최저 점수를 비교하여 더 작은 쪽을 beta로 결정한다.
            beta = min(beta, value)

          ## 만약 MIN의 현재 선택 이후 다른 MIN이 내린 최고의 선택이 MIN의 다른 선택에서 다른 MIN이 내린 최고의 선택만 못하다면, 이후의 선택은 고려할 필요가 없기 때문에 바로 반환한다.
          if value < alpha:
            return (value, move)

      ## MAX의 다음 움직임을 반환한다.
      return (value, move)
    
    ## alpha: 탐색 과정에서 여태까지 MAX가 내린 선택의 결과 중 최선을 alpha라 부른다.
    alpha=float("-inf")

    ## beta: 탐색 과정에서 여태까지 MIN이 내린 선택의 결과 중 최선을 beta라 부른다.
    beta=float("inf")

    ## Alpha-Beta Pruning을 적용한 Minimax 알고리즘을 통해 MAX가 가장 유리한 선택지를 계산한다.
    (score, move) = max_agent_evaluationFunc(self, gameState, self.depth, alpha, beta)

    ## MAX의 다음 움직임을 반환한다.
    return move
    ############################################################################



class ExpectimaxAgent(AdversialSearchAgent):
  """
    [문제 03] ExpectimaxAgent의 Action을 구현하시오.
    (depth와 evaluation function은 위에서 정의한 self.depth and self.evaluationFunction을 사용할 것.)
  """
  def Action(self, gameState):
    ####################### Write Your Code Here ################################

    ## MAX가 가장 유리한 선택지를 계산한다.
    def max_agent_evaluationFunc(self, currentGameState, depth):

      ## 정해진 깊이에 도달하거나 게임 오버 상황에 처한 경우 앞으로의 탐색은 하지 않는다.
      if depth == 0 or currentGameState.isWin() or currentGameState.isLose():

        ## 이때 계산한 점수와 움직임을 반환한다.
        return (self.evaluationFunction(currentGameState), None)
      
      ## value: MAX가 가장 유리한 선택을 했을 때의 점수.
      value = float("-inf")

      ## exp_value: 
      exp_value = 0

      ## move: MAX의 가장 유리한 선택
      move = Directions.STOP

      #현재 상태에서 MAX가 내릴 수 있는 모든 선택에 대하여 그 이후의 상태를 계산한다.
      for action in currentGameState.getLegalActions():

        ## successorGameState: 현재 선택을 실행한 이후의 상태.
        successorGameState = currentGameState.generateSuccessor(0, action)

        ##expectimax 알고리즘을 통하여 가장 높은 기대갑을 가지는 선택을 내린다.
        (tmp, tmp_move) = exp_agent_evaluationFunc(self, successorGameState, depth, 1)

        if tmp > value:
          value = tmp
          move = action

        ## 만약 tmp가 기존의 최고값보다 크다면, 현재의 선택 action이 기존의 선택보다 낫다는 뜻이기 때문에 점수를 갬신하고 움직임을 action으로 바꾼다.
        exp_value += tmp

      ## MAX의 다음 움직임을 반환한다.
      return (exp_value/len(currentGameState.getLegalActions()), move)
  
    ## 확률을 고려하여 각 선택이 가질 수 있는 기대값을 계산하고, 이에 기반하여 최선의 선택을 결정한다.
    def exp_agent_evaluationFunc(self, currentGameState, depth, agentIndex):

      ## 정해진 깊이에 도달하거나 게임 오버 상황에 처한 경우 앞으로의 탐색은 하지 않는다.
      if depth == 0 or currentGameState.isWin() or currentGameState.isLose():

        ## 이때 계산한 점수와 움직임을 반환한다.
        return (self.evaluationFunction(currentGameState), None)
      
      ## exp_value: 모든 선택을 고려한 기대값.
      exp_value = 0

      ## value: MAX가 가장 유리한 선택을 했을 때의 점수.
      value = float("inf")

      ## move: 기대값이 가장 높은 선택.
      move = Directions.STOP

      ## 만약 마지막 MIN (팩맨 게임에서는 유령)을 고려하고 있다면, 다음에는 깊이를 한 단계 더 늘리고 MAX의 다음 움직임을 고려해 마지막 MIN의 선택을 결정한다.
      if agentIndex == currentGameState.getNumAgents()-1:

        ## 현재의 MIN이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
        for action in currentGameState.getLegalActions(agentIndex):

          ## successorGameState: 현재 선택을 실행한 이후의 상태.
          successorGameState = currentGameState.generateSuccessor(agentIndex, action)

          ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 최선의 움직임은 tmp_move.
          (tmp, tmp_move) = max_agent_evaluationFunc(self, successorGameState, depth-1)

          if tmp > value:
            value = tmp
            move = action

          ## 기대값을 더한다.
          exp_value += tmp

      ## 아직 모든 MIN(팩맨 게임의 유령)을 고려하지 않았다면 depth를 유지하고 다음 MIN의 선택을 고려한다.
      else:

        ## 현재의 MIN이 내릴 수 있는 모든 선택지 중에서 가장 득이 되는 선택지를 계산한다.
        for action in currentGameState.getLegalActions(agentIndex):

          ## successorGameState: 현재 선택을 실행한 이후의 상태.
          successorGameState = currentGameState.generateSuccessor(agentIndex, action)

          ## 그때 이후 발생할 수 있는 모든 상황을 고려했을 때, 최선의 선택은 tmp, 최선의 움직임은 tmp_move.
          (tmp, tmp_move) = exp_agent_evaluationFunc(self, successorGameState, depth, agentIndex+1)

          if tmp < value:
            value = tmp
            move = action

          ## 기대값을 더한다.
          exp_value += tmp

      ## 모든 선택지에 대한 기대값을 계산한 후, 선택지의 개수로 나누어 평균 기대값을 구한다.
      ## 모든 선택은 동일한 확률을 가지기 때문에, 기대값을 곱할 필요 없이 일괄적으로 더한 후 선택지의 개수로 나누어 평균 기대값을 구했다.
      return (exp_value/len(currentGameState.getLegalActions(agentIndex)), move)
    
    ## Expectimax 알고리즘을 통해 MAX가 가장 유리한 선택지를 계산한다.
    (score, move) = max_agent_evaluationFunc(self, gameState, self.depth)

    ## MAX의 다음 움직임을 반환한다.
    return move

    ############################################################################
