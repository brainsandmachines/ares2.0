import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_training_plots(csv_path):
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    df = pd.read_csv(csv_path)

    # Filter only rows with numeric epoch
    df_filtered = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()].copy()
    df_filtered["epoch"] = df_filtered["epoch"].astype(int)

    # Remove duplicate epochs — keep last
    df_unique = (
        df_filtered.sort_values("epoch")
                   .groupby("epoch")
                   .tail(1)
                   .reset_index(drop=True)
    )
    df_unique = df_unique.apply(pd.to_numeric, errors="coerce")
    saved_plots = {}

    for m in df_unique.columns:
        if m != "epoch":
            # Convert metric to numeric (critical fix!)
            plot_dir = os.path.join(out_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)

            plt.figure()
            plt.plot(df_unique["epoch"], df_unique[m])
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.title(f"{m} vs Epoch")
            plt.grid(True)

            fig_path = os.path.join(plot_dir, f"{m}.png")
            plt.savefig(fig_path)
            plt.close()
            saved_plots[m] = fig_path

    return saved_plots

if __name__ == "__main__":
    for root, dirs, files in os.walk("/home/ashtomer/projects/ares/results"):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                print(f"Generating plots for {csv_file_path}")
                plots = generate_training_plots(csv_file_path)
                for metric, plot_path in plots.items():
                    print(f"Saved plot for {metric} at {plot_path}")
