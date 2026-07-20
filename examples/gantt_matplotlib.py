import numpy as np

from maxplotlib import Canvas


def main() -> None:
    # Define project tasks
    tasks = [
        "Planning",
        "Design",
        "Development",
        "Testing",
        "Deployment",
        "Documentation",
    ]
    
    # Start times (in days from project start)
    start_times = np.array([0, 5, 10, 25, 35, 30])
    
    # Duration of each task (in days)
    durations = np.array([5, 5, 15, 10, 5, 10])
    
    # Create canvas
    canvas = Canvas(width="14cm", ratio=0.6, dpi=150)
    
    # Add gantt chart
    canvas.gantt(
        tasks=tasks,
        start_times=start_times,
        durations=durations,
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
        label="Project Tasks"
    )
    
    # Configure plot
    canvas.set_title("Project Timeline - Gantt Chart")
    canvas.set_xlabel("Days from Project Start")
    canvas.set_ylabel("Tasks")
    canvas.set_grid(True)
    canvas.set_xlim(0, 45)
    
    # Save figure
    canvas.savefig("gantt_matplotlib.png", backend="matplotlib")
    print("Gantt chart saved as gantt_matplotlib.png")


if __name__ == "__main__":
    main()
