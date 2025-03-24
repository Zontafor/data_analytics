import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Simulate departmental workload data
np.random.seed(42)
departments = ['Data Engineering', 'Analytics', 'QA', 'UI/UX', 'Project Management']
months = ['Jan', 'Feb', 'Mar', 'Apr']

def simulate_workload(base, increment, noise=3):
    return [min(100, base + i * increment + np.random.normal(0, noise)) for i in range(4)]

workload_data = {
    department: simulate_workload(base=55 + i * 5, increment=10) for i, department in enumerate(departments)
}
df = pd.DataFrame(workload_data, index=months).T

# Generate opacity scaling
opacities = np.linspace(0.6, 1.0, len(df))

# Create enhanced contrast heatmap
fig, ax = plt.subplots(figsize=(10, 6))
heatmap = sns.heatmap(df, annot=True, fmt='.0f', cmap='OrRd', vmin=0, vmax=100, cbar_kws={'label': 'Workload (%)'}, ax=ax)
heatmap.invert_yaxis()
heatmap.set_xlabel("")
heatmap.set_ylabel("")

# Apply opacity overlays
for y, (dept, row) in enumerate(df.iterrows()):
    for x, value in enumerate(row):
        heatmap.add_patch(plt.Rectangle((x, y), 1, 1, color='white', alpha=1 - opacities[y], linewidth=0))

plt.title('Simulated Departmental Workload vs Capacity (Enhanced Contrast)')
plt.tight_layout()
plt.savefig("simulated_departmental_workload_contrast.png")
plt.show()