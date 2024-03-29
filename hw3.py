import copy, queue

def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" and "someone" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    # -*- coding: utf-8 -*-
    # standardized_variables : 규칙을 인자로 받고 이 규칙에 따라 모든 변수를 고유한 이름으로 바꿔주는 함수.

    # standardized_rules : standardized_variables를 사용하여 규칙에 따라 모든 변수를 고유한 이름으로 바꾼다.
    standardized_rules = copy.deepcopy(nonstandard_rules)

    # -*- coding: utf-8 -*-
    #variables : 모든 변수를 저장하는 집합.
    variables = set()

    for rule in standardized_rules.values():
        substitutes = []
        for antecedents in rule["antecedents"]:
          substitute = []
          # 변수 이름: 
          # ax : someone in the antecedent
          # ay : something in the antecedent
          # cx : someone in the consequent
          # cy : something in the consequent
          # + len(variables) : 고유한 이름을 위해 변수의 개수를 더해준다.
          for antecedent in antecedents:
              if antecedent == "someone":
                  substitute.append("ax" + str(len(variables)))
                  rule["antecedents"] = "ax" + str(len(variables))
                  variables.add("ax" + str(len(variables)))
              elif antecedent == "something":
                  substitute.append("ay" + str(len(variables)))
                  rule["antecedents"] = "ay" + str(len(variables))
                  variables.add("ay" + str(len(variables)))
              else:
                  substitute.append(antecedent)
          substitutes.append(substitute)
        rule.update({"antecedents": substitutes})
        substitute = []
        for consequent in rule["consequent"]:
            if consequent == "someone":
                substitute.append("cx" + str(len(variables)))
                rule["consequent"][0] = "cx" + str(len(variables))
                variables.add("cx" + str(len(variables)))
            elif consequent == "something":
                substitute.append("cy" + str(len(variables)))
                rule["consequent"][0] = "cy" + str(len(variables))
                variables.add("cy" + str(len(variables)))
            else:
                substitute.append(consequent)
        rule.update({"consequent": substitute})
    
    return standardized_rules, list(variables)

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'}
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    
    subs = {}
    input_query = copy.deepcopy(query)
    unification = copy.deepcopy(query)
    input_datum = copy.deepcopy(datum)
    input_variables = copy.deepcopy(variables)

    
    # 만약 query와 datum이 같다면, unification은 필요없다.
    if query == datum:
        return query, subs
    # query와 datum의 길이가 다르다면, unification은 필요없다.
    elif query == []:
        return datum, subs
    elif datum == []:
        return query, subs
    # 만약 query와 datum의 진릿값이 다르다면, unification은 필요없다.
    elif query[-1] is True and datum[-1] is False or query[-1] is False and datum[-1] is True:
        return None, None
    
    # query와 datum의 길이가 같다면, unification이 필요하다.
    for i in range(len(query)):
        # query와 datum의 i번째 원소가 같다면, substitution이 필요없다.
        if input_query[i] != input_datum[i]:
            if input_query[i] in input_variables:
                subs[input_query[i]] = input_datum[i]
            elif input_datum[i] in input_variables and input_query[i] not in input_variables:
                subs[input_datum[i]] = input_query[i]
            elif input_query[i] not in input_variables and input_datum[i] not in input_variables:
                return None, None
        for comp in unification:
            if comp == input_query[i]:
                comp = input_datum[i]
            elif comp == input_datum[i]:
                comp = input_query[i]

    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
        => goals에서 "bobcat eats squirrel, False"를 지우고 application['antecedents']를 추가.
        => application을 먼저 만들고, 여기서 goalsets를 만들어야 할듯...
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''

    input_rule = copy.deepcopy(rule)
    input_goals = []
    input_goals.append(copy.deepcopy(goals))
    input_variables = copy.deepcopy(variables)
    applications = []
    goalsets = []
    #print("\napply_rule")
    #print(input_rule)
    #print("\napply_goals")
    #print(input_goals)
    #print("\napply_variables")
    #print(input_variables)
    #print("=========================")
    
    # A->B, B => A
    # if query in rule...return
    # if there is no query left, only propositions with constants...return

    for goal in input_goals:
      unification, subs = unify(goal, input_rule['consequent'], input_variables)
      if unification is not None:
        tmp_rl = []
        tmp_ap = copy.deepcopy(rule)
        if input_rule['antecedents'] is not None:
            for antecedent in input_rule['antecedents']:
                sub1, sub2 = unify(goal, antecedent, input_variables)
                tmp_rl.append(sub1) 
        tmp_ap['antecedents'] = tmp_rl
        tmp_ap['consequent'] = unification
        tmp_gl = copy.deepcopy(goals)
        tmp_gl.append(tmp_rl)
        applications.append(tmp_ap)
        goalsets.append(tmp_gl)

    return applications, goalsets

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables

    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    
    proof = []
    goals = []
    if query in rules.values():
        proof.append(query)
        return proof
    for rule in rules.values():
        application, goals = apply(rule, goals, variables)
        proof = goals

    return proof
