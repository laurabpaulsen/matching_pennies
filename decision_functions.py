import numpy as np


# Definintion of different decision functions
def random_decision(bias):
    return np.random.choice([0, 1], p=[bias, 1-bias])

def winstay_looseswitch(history:list[int], feedback:list[bool]):
    # if no choices have been made, return random decision
    if len(history) == 0:
        return random_decision(0.5)
    
    if feedback[-1]:
        return history[-1]
    else:
        return 1 if history[-1] == 0 else 0



def perfect_memory(history:list[int], feedback:list[bool], memory_window = False):
    # if no choices have been made, return random decision
    if len(history) == 0:
        return random_decision(0.5)
    
    if memory_window:
        if memory_window < len(history):
            history = history[-memory_window:]
            feedback = feedback[-memory_window:]


    idx_0 = [i for i in range(len(history)) if history[i] == 0]
    idx_1 = [i for i in range(len(history)) if history[i] == 1]

    # get the sum of postive feedback for both choices
    sum_0 = np.sum([1 if feedback[i] == True else -1 for i in idx_0])
    sum_1 = np.sum([1 if feedback[i] == True else -1 for i in idx_1])

    if int(sum_0) == int(sum_1):
        return random_decision(0.5)

    elif sum_0 < sum_1:
        return 1
    
    else:
        return 0