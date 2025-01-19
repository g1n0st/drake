import json
import os
import matplotlib.pyplot as plt

def get_info(dir, big_dt):
    json_files = [file for file in os.listdir(dir) if file.endswith('.json')]

    total_iterations = 0
    max_iterations = 0
    total_time = 0.0

    for json_file in json_files:
        file_path = os.path.join(dir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            total_iterations += len(data)
            max_iterations = max(max_iterations, len(data))
            total_time += sum(entry.get("time", 0.0) for entry in data if isinstance(entry, dict))

    return (max_iterations, total_iterations, total_time, len(json_files), big_dt)

def plot(axes, y, color, label):
    # Extract metrics for plotting
    max_iterations = [info[0] for info in y]
    total_iterations = [info[1] for info in y]
    average_iterations = [info[1] / info[3] if info[3] > 0 else 0 for info in y]
    total_time = [info[2] / 1e3 for info in y]
    x = [info[4] for info in y]

    axes[0, 0].plot(x, max_iterations, marker='o', color=color, alpha=0.5, label=label, linewidth=4, markersize=10)
    axes[0, 1].plot(x, total_iterations, marker='o', color=color, alpha=0.5, label=label, linewidth=4, markersize=10)
    axes[1, 0].plot(x, average_iterations, marker='o', color=color, alpha=0.5, label=label, linewidth=4, markersize=10)
    axes[1, 1].plot(x, total_time, marker='o', color=color, alpha=0.5, label=label, linewidth=4, markersize=10)

back_1e_3 = [get_info(f"/home/changyu/drake/mpm-data/small_1e-3_big_{i+1}e-3", (i+1) * 1e-3) for i in range(10)]
# back_1e_4 = [get_info(f"./small_1e-4_big_{i+1}e-3", (i+1) * 1e-3) for i in range(9)]
exact_1e_3 = [get_info(f"/home/changyu/drake/mpm-data/exact_small_1e-3_big_{i+1}e-3", (i+1) * 1e-3) for i in range(10)]
# exact_1e_4 = [get_info(f"./exact_small_1e-4_big_{i+1}e-3", (i+1) * 1e-3) for i in range(9)]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.rcParams.update({'font.size': 14})

plot(axes, back_1e_3, "blue", "substep(small) dt=1e-3, backtracking")
# plot(axes, back_1e_4, "red", "substep(small) dt=1e-4, backtracking")
plot(axes, exact_1e_3, "purple", "substep(small) dt=1e-3, exact")
# plot(axes, exact_1e_4, "yellow", "substep(small) dt=1e-4, exact")

axes[0, 0].set_xlabel('step(big) dt')
axes[0, 0].set_ylabel('max newton iterations within 0.5s')
axes[0, 0].legend()

axes[0, 1].set_xlabel('step(big) dt')
axes[0, 1].set_ylabel('total newton iterations within 0.5s')
axes[0, 1].legend()

axes[1, 0].set_xlabel('step(big) dt')
axes[1, 0].set_ylabel('average newton iterations per step')
axes[1, 0].legend()

axes[1, 1].set_xlabel('step(big) dt')
axes[1, 1].set_ylabel('total time (s)')
axes[1, 1].legend()

import matplotlib.ticker as ticker
def format_scientific(x, _):
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e+")

for ax in axes.flatten():
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_scientific))

plt.tight_layout()
# plt.subplots_adjust(hspace=0, wspace=0.3)
plt.savefig("/home/changyu/drake/mpm-data/plot.pdf")