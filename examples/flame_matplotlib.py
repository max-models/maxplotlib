"""
Example: Flame Chart with Matplotlib Backend

This example demonstrates how to create a flame chart visualization
using the matplotlib backend. Flame charts are useful for visualizing
hierarchical profiling data, showing function call stacks and their
execution times.
"""

from maxplotlib import Canvas

# Example profiling data: function call hierarchy
# Each function has: label, parent index (None for root), and duration
labels = [
    "main()",  # 0 - root
    "process_data()",  # 1 - child of main
    "load_file()",  # 2 - child of process_data
    "parse_json()",  # 3 - child of process_data
    "validate()",  # 4 - child of process_data
    "compute()",  # 5 - child of main
    "algorithm_a()",  # 6 - child of compute
    "algorithm_b()",  # 7 - child of compute
    "save_results()",  # 8 - child of main
]

parents = [
    None,  # main() is root
    0,  # process_data() called by main()
    1,  # load_file() called by process_data()
    1,  # parse_json() called by process_data()
    1,  # validate() called by process_data()
    0,  # compute() called by main()
    5,  # algorithm_a() called by compute()
    5,  # algorithm_b() called by compute()
    0,  # save_results() called by main()
]

# Duration of each function call (in milliseconds)
values = [100, 40, 10, 15, 15, 50, 25, 25, 10]

# Start times for each function (when they begin execution)
start_times = [0, 0, 0, 10, 25, 40, 40, 65, 90]

# Create canvas and add flame chart
canvas = Canvas(nrows=1, ncols=1, figsize=(12, 6))
canvas.flame_chart(
    labels=labels,
    parents=parents,
    values=values,
    start_times=start_times,
    colormap="viridis",
    edgecolor="black",
)

# Configure the plot
canvas.set_xlabel("Time (ms)")
canvas.set_ylabel("Stack Depth")
canvas.set_title("Flame Chart: Function Call Hierarchy")

# Save the figure
canvas.savefig("flame_matplotlib.png", backend="matplotlib")
print("Flame chart saved as flame_matplotlib.png")
