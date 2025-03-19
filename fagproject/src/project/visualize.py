# Re-import necessary libraries after execution state reset
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.linspace(0, 10, 20)  # 20 points
y = np.random.rand(20)
sizes = np.random.rand(20) * 1000  # 20 varying sizes
shapes = np.array(["x"] * 10 + ["o"] * 10)  # 10 'x' and 10 'o'
np.random.shuffle(shapes)  # Shuffle the shapes

# Define three distinct colors
colors = np.array(["red", "blue", "green"])
point_colors = np.random.choice(colors, 20)  # Assign one of the three colors to each point

# Create the plot
plt.figure(figsize=(8, 6))
for i in range(len(x)):
    plt.scatter(x[i], y[i], s=sizes[i], marker=shapes[i], color=point_colors[i])

# Set plot labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Random Plot with Varying Sizes, Shapes, and Three Colors')

# Show plot
plt.show()