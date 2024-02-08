from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt


def plot_proportion_correct(data:pd.DataFrame, savepath:Path = None, column = "bias"):

    data["correct"] = data["feedback"].astype(int)

    fig, ax = plt.subplots(1, figsize=(10, 7), dpi = 300)

    for var in data[column].unique():
        tmp_data = data[data[column] == var].copy()
        tmp_data["correct"] = tmp_data["correct"].cumsum() / (tmp_data.index + 1)
        ax.plot(tmp_data.index, tmp_data["correct"], label=var)


    ax.set_xlabel("Trials")
    ax.set_ylabel("Proportion correct")
    ax.legend(title=column, loc="upper right", ncol=2)

    if savepath:
        plt.savefig(savepath)
    

if __name__ in "__main__":
    path = Path(__file__).parent
    inpath = path / "data"

    fig_path = path / "figures"
    if not fig_path.exists():
        fig_path.mkdir(parents=True)


    data = pd.DataFrame()
    for f in inpath.iterdir():
        if "bias" not in str(f):
            continue
        data_tmp = pd.read_csv(f)

        bias = str(f).split("_")[-1].split(".csv")[0]
        data_tmp["bias"] = bias
        data = pd.concat([data, data_tmp])

    plot_proportion_correct(data, fig_path / "proportion_correct_perfect_memory_vs_different_bias.png")


    data = pd.DataFrame()
    for f in inpath.iterdir():
        if "window" not in str(f):
            continue
        
        data_tmp = pd.read_csv(f)

        window = str(f).split("_")[-1].split(".csv")[0]
        data_tmp["window"] = window
        data = pd.concat([data, data_tmp])

    plot_proportion_correct(
        data, 
        fig_path / "proportion_correct_bias_0.7_vs_different_window.png",
        column = "window")