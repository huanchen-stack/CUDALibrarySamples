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

for model in ["qwen", "mixtral"]:

    # if model == "mixtral":
    #     nvfp4_times = [0.031971, 0.026903, 0.017878, 0.018286, 0.038200, 0.051696, 0.089943]
    #     mxfp8_times = [0.057560, 0.030071, 0.029754, 0.030085, 0.069115, 0.107542, 0.187313]
    #     fp8_times = [0.013490, 0.017457, 0.017485, 0.030701, 0.055265, 0.098988, 0.173424]

    # if model == "qwen":
    #     nvfp4_times = [0.010862, 0.010794, 0.010810, 0.010951, 0.011201, 0.020169, 0.029917]
    #     mxfp8_times = [0.017219, 0.016985, 0.017005, 0.017040, 0.017264, 0.036774, 0.056450]
    #     fp8_times = [0.006182, 0.008189, 0.008189, 0.011300, 0.018424, 0.031748, 0.051740]

    if model == "qwen":
        nvfp4_times = [
            0.042220,  # batch = 1
            0.042683,  # batch = 32
            0.042862,  # batch = 64
            0.043068,  # batch = 128
            0.043926,  # batch = 256
            0.079073,  # batch = 512
            0.116238,  # batch = 1024
        ]
        mxfp8_times = [
            0.067496,  # batch = 1
            0.067925,  # batch = 32
            0.067834,  # batch = 64
            0.067908,  # batch = 128
            0.068388,  # batch = 256
            0.129466,  # batch = 512
            0.190303,  # batch = 1024
        ]
        fp8_times = [
            0.018423,  # batch = 1
            0.022540,  # batch = 32
            0.024561,  # batch = 64
            0.036863,  # batch = 128
            0.065321,  # batch = 256
            0.118811,  # batch = 512
            0.165730,  # batch = 1024
        ]
        hsh_times = [
            0.046966,  # batch = 1
            0.059032,  # batch = 32
            0.059293,  # batch = 64
            0.106707,  # batch = 128
            0.198719,  # batch = 256
            0.311113,  # batch = 512
            0.532647,  # batch = 1024
        ]
    if model == "mixtral":
        nvfp4_times = [
            0.068125,  # batch = 1
            0.068475,  # batch = 32
            0.070392,  # batch = 64
            0.071810,  # batch = 128
            0.140300,  # batch = 256
            0.195825,  # batch = 512
            0.336041,  # batch = 1024
        ]
        mxfp8_times = [
            0.117055,  # batch = 1
            0.119070,  # batch = 32
            0.118220,  # batch = 64
            0.118907,  # batch = 128
            0.232265,  # batch = 256
            0.355988,  # batch = 512
            0.652584,  # batch = 1024
        ]
        fp8_times = [
            0.040939,  # batch = 1
            0.061407,  # batch = 32
            0.061461,  # batch = 64
            0.112641,  # batch = 128
            0.212237,  # batch = 256
            0.315453,  # batch = 512
            0.567882,  # batch = 1024
        ]
        hsh_times = [
            0.156413,  # batch = 1
            0.202810,  # batch = 32
            0.387537,  # batch = 64
            0.389244,  # batch = 128
            0.616467,  # batch = 256
            1.097616,  # batch = 512
            2.093923,  # batch = 1024
        ]


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
    # ax.plot(x_positions, hsh_times, '-o', color='grey', linewidth=1.5, 
    #         markersize=4, label='HSH')

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
        plt.savefig('precisions_nvfp4_mxfp8_fp8_mixtral_moe.png', dpi=300, bbox_inches='tight')
    if model == "qwen":
        plt.savefig('precisions_nvfp4_mxfp8_fp8_qwen_moe.png', dpi=300, bbox_inches='tight')
