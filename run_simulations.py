from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
from agent import Agent, game
from decision_functions import *


def plot_proportion_correct(data:pd.DataFrame, savepath:Path = None, column = "bias", title:str = "Proportion correct"):

    data["correct"] = data["feedback"].astype(int)

    fig, ax = plt.subplots(1, figsize=(10, 7), dpi = 300)

    for var in data[column].unique():
        tmp_data = data[data[column] == var].copy()
        tmp_data["correct"] = tmp_data["correct"].cumsum() / (tmp_data.index + 1)
        ax.plot(tmp_data.index, tmp_data["correct"], label=var)


    ax.set_xlabel("Trials")
    ax.set_ylabel("Proportion correct")
    ax.legend(title=column, loc="upper right", ncol=2)

    ax.set_title(title)

    if savepath:
        plt.savefig(savepath)
    
def plot_choice(data:pd.DataFrame, savepath:Path = None, hider_col = "hider", player_col = "player", column = "bias", title:str = "Agent Choices"):

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), dpi = 300)

    for var in data[column].unique():
        tmp_data = data[data[column] == var].copy()

        # choices can be 0 or 1
        # plot the proportion of 1s
        cum_sum_player = tmp_data[player_col].astype(int)
        cum_sum_player = tmp_data[player_col].cumsum() / (tmp_data["trial"] + 1)

        cum_sum_hider = tmp_data[hider_col].astype(int)
        cum_sum_hider = tmp_data[hider_col].cumsum() / (tmp_data["trial"] + 1)

        axes[0].plot(tmp_data["trial"], cum_sum_player, label=var)
        axes[1].plot(tmp_data["trial"], cum_sum_hider, label=var)
    
    axes[0].set_title("Player choice")
    axes[0].set_xlabel("Trials")
    axes[0].set_ylabel("Proportion of 1s")
    axes[0].legend(title=column, loc="upper right", ncol=2)


    axes[1].set_title("Hider choice")
    axes[1].set_xlabel("Trials")
    axes[1].set_ylabel("Proportion of 1s")
    axes[1].legend(title=column, loc="upper right", ncol=2)

    fig.suptitle(title)

    if savepath:
        plt.savefig(savepath)
    

def vary_bias(bias_values:list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
    """
    Runs the matching pennies game with a perfect memory hider and a random decision player with different bias values.

    Parameters
    ----------
    bias_values : list, optional
        The bias values to test. The default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].
    
    Returns
    -------
    data : pd.DataFrame
        The results of the game.
    """
    data = pd.DataFrame()

    for bias in bias_values:
        hider = Agent(perfect_memory, include_history=True)
        player = Agent(random_decision, bias = 0.5)
        
        data_tmp = game(player, hider, 120)
        
        data_tmp["bias"] = bias
        data = pd.concat([data, data_tmp])

    return data

def vary_window(window_values:list = [1, 2, 5, 10, 20, 30, 40, 50, None]):
    """
    Runs the matching pennies game with a perfect memory hider (with different window sizes) and a random decision player with a bias of 0.7.

    Parameters
    ----------
    window_values : list, optional
        The window sizes to test. The default is [1, 2, 5, 10, 20, 30, 40, 50, None].
    
    Returns
    -------
    data : pd.DataFrame
        The results of the game.
    """
    data = pd.DataFrame()

    for window_size in window_values:
        hider = Agent(perfect_memory, include_history=True, memory_window = window_size)
        player = Agent(random_decision, bias = 0.7)
        data_tmp = game(player, hider, 120)

        data_tmp["window"] = window_size
        data = pd.concat([data, data_tmp])

    return data

if __name__ in "__main__":
    path = Path(__file__).parent
    inpath = path / "data"

    fig_path = path / "figures"
    if not fig_path.exists():
        fig_path.mkdir(parents=True)


    # run simulations with perfect memory hider and random decision player with different bias values
    data_bias = vary_bias()

    plot_proportion_correct(
        data_bias, 
        fig_path / "proportion_correct_perfect_memory_vs_different_bias.png",
        column = "bias",
        title="Hider perfect memory, player random decision with different bias values"
        )
    
    plot_choice(
        data_bias, 
        fig_path / "player_choice_perfect_memory_vs_different_bias.png",
        column = "bias",
        title="Hider perfect memory, player random decision with different bias values"
        )


    # run simulations with perfect memory hider (with different window sizes) and random decision player with a bias of 0.7
    data_window = vary_window()

    plot_proportion_correct(
        data_window, 
        fig_path / "proportion_correct_bias_0.7_vs_different_window.png",
        column = "window",
        title="Player bias = 0.7, Hider perfect memory with different window sizes"
        )
    
    plot_choice(
        data_window, 
        fig_path / "player_choice_bias_0.7_vs_different_window.png",
        column = "window",
        title="Player bias = 0.7, Hider perfect memory with different window sizes"
        )
    
    

