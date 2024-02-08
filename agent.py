import numpy as np
import pandas as pd
from decision_functions import *
from pathlib import Path

class Agent:
    def __init__(self, decision_function, include_history = False, **kwargs):
        """
        Parameters
        ----------
        decision_function : function
            The decision function that the agent will use to make decisions.
        include_history : bool, optional
            Whether to include the history of the game in the decision function. The default is False.
        """
        self.decision_function = decision_function
        self.kwargs = kwargs
        self.include_history = include_history

        self.history = []
        self.feedback = []

    def decision(self):
        if self.include_history:
            return self.decision_function(self.history, self.feedback, **self.kwargs)
        else:
            return self.decision_function(**self.kwargs)
        
    def update_history(self, move, feedback):
        self.history.append(move)
        self.feedback.append(feedback)



def game(player, hider, trials):

    game_data = pd.DataFrame(columns=["hider", "player", "feedback"])

    for t in range(trials):
        hider_move = hider.decision()
        player_move = player.decision()

        feedback = hider_move == player_move

        player.update_history(player_move, feedback)
        hider.update_history(hider_move, hider_move != player_move)
        
        game_data = pd.concat([game_data, pd.DataFrame({"hider": hider_move, "player": player_move, "feedback": feedback}, index=[t])])
    
    return game_data



if __name__ in "__main__":

    path = Path(__file__).parent
    outpath = path / "data"

    if not outpath.exists():
        outpath.mkdir(parents=True)


    for bias in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        hider = Agent(perfect_memory, include_history=True)
        player = Agent(random_decision, bias = 0.5)
        data = game(player, hider, 120)

        data.to_csv(outpath / f"bias_{bias}.csv")

    for window_size in [1, 2, 5, 10, 20, 30, 40, 50, None]:
        hider = Agent(perfect_memory, include_history=True, memory_window = window_size)
        player = Agent(random_decision, bias = 0.7)
        data = game(player, hider, 120)

        data.to_csv(outpath / f"window_{window_size}.csv")



