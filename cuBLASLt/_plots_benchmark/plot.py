import matplotlib.pyplot as plt
import numpy as np

# mixtral
# batch_sizes = [1, 32, 64, 128, 256, 512, 1024, 2048]
# model = "qwen"

# if model == "mixtral":
#     nvfp4_times = [0.031971, 0.026903, 0.017878, 0.018286, 0.038200, 0.051696, 0.089943, 0.176869]
#     mxfp8_times = [0.057560, 0.030071, 0.029754, 0.030085, 0.069115, 0.107542, 0.187313, 0.377123]
#     fp8_times = [0.013490, 0.017457, 0.017485, 0.030701, 0.055265, 0.098988, 0.173424, 0.280313]

# if model == "qwen":
#     nvfp4_times = [0.010862, 0.010794, 0.010810, 0.010951, 0.011201, 0.020169, 0.029917, 0.048987]
#     mxfp8_times = [0.017219, 0.016985, 0.017005, 0.017040, 0.017264, 0.036774, 0.056450, 0.095722]
#     fp8_times = [0.006182, 0.008189, 0.008189, 0.011300, 0.018424, 0.031748, 0.051740, 0.087413]

batch_sizes = [1, 32, 64, 128, 256, 512, 1024]
model = "qwen"

if model == "mixtral":
    nvfp4_times = [0.031971, 0.026903, 0.017878, 0.018286, 0.038200, 0.051696, 0.089943]
    mxfp8_times = [0.057560, 0.030071, 0.029754, 0.030085, 0.069115, 0.107542, 0.187313]
    fp8_times = [0.013490, 0.017457, 0.017485, 0.030701, 0.055265, 0.098988, 0.173424]

if model == "qwen":
    nvfp4_times = [0.010862, 0.010794, 0.010810, 0.010951, 0.011201, 0.020169, 0.029917]
    mxfp8_times = [0.017219, 0.016985, 0.017005, 0.017040, 0.017264, 0.036774, 0.056450]
    fp8_times = [0.006182, 0.008189, 0.008189, 0.011300, 0.018424, 0.031748, 0.051740]


# Create figure with compact size
fig, ax = plt.subplots(figsize=(6, 4))

# Plot lines with evenly spaced x-positions
x_positions = np.arange(len(batch_sizes))
ax.plot(x_positions, mxfp8_times, 'o-', color='green', linewidth=1.5, 
        markersize=4, label='MXFP8')
ax.plot(x_positions, nvfp4_times, 's-', color='red', linewidth=1.5, 
        markersize=4, label='NVFP4')
ax.plot(x_positions, fp8_times, '^-', color='blue', linewidth=1.5, 
        markersize=4, label='FP8')

# Labels and formatting
ax.set_xlabel('Batch Size', fontsize=9)
ax.set_ylabel('Average Runtime (ms)', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(fontsize=8, frameon=True, loc='upper left')

# Set x-ticks to actual batch sizes
ax.set_xticks(x_positions)
ax.set_xticklabels(batch_sizes)

# Set y-axis to start from 0
ax.set_ylim(bottom=0)

# Tight layout
plt.tight_layout(pad=0.3)

# Save figure
if model == "mixtral":
    plt.savefig('nvfp4_mxfp8_mixtral_moe.png', dpi=300, bbox_inches='tight')
if model == "qwen":
    plt.savefig('nvfp4_mxfp8_qwen_moe.png', dpi=300, bbox_inches='tight')
