import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulate task completion data
np.random.seed(42)
tasks = ['Design', 'Development', 'QA Testing', 'Documentation', 'Deployment']
weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']

def simulate_completion(start_mean, growth_rate, noise=5):
    return [min(max(0, start_mean + growth_rate * week + np.random.normal(0, noise)), 100) for week in range(4)]

completion_data = {
    task: simulate_completion(start_mean=10 + i * 5, growth_rate=20) for i, task in enumerate(tasks)
}
df = pd.DataFrame(completion_data, index=weeks).T

# Generate opacity scaling
opacities = np.linspace(0.6, 1.0, len(df))

# Create enhanced contrast heatmap
fig, ax = plt.subplots(figsize=(10, 6))
heatmap = sns.heatmap(df, annot=True, fmt='.0f', cmap='YlGnBu', vmin=0, vmax=100, cbar_kws={'label': 'Completion %'}, ax=ax)
heatmap.invert_yaxis()
heatmap.set_xlabel("")
heatmap.set_ylabel("")

# Apply opacity overlays
for y, (task, row) in enumerate(df.iterrows()):
    for x, value in enumerate(row):
        heatmap.add_patch(plt.Rectangle((x, y), 1, 1, color='white', alpha=1 - opacities[y], linewidth=0))

plt.title('Simulated Task Completion Status by Week (Enhanced Contrast)')
plt.tight_layout()
plt.savefig("simulated_task_completion_contrast.png")
plt.show()