import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_kd_val_loss():
    epochs = np.arange(1, 51)
    val_losses_200_student =    [0.447684, 0.445861, 0.445861, 0.444829, 0.444555, 0.444392, 0.444350, 0.444215, 0.444146, 0.444105, 0.444037, 0.443953, 0.443861, 0.443873, 0.443826, 0.443804, 0.443762, 0.443824,
                                 0.443779, 0.443809, 0.443819, 0.443730, 0.443752, 0.443747, 0.443667, 0.443702, 0.443717, 0.443705, 0.443660, 0.443651, 0.443660, 0.443659, 0.443644, 0.443574, 0.443631, 0.443617,
                                 0.443642, 0.443607, 0.443639] + [None] * 11
    val_losses_200_small =      [0.447920, 0.445893, 0.445157, 0.444787, 0.444532, 0.444369, 0.444199, 0.444167, 0.444059, 0.444021, 0.443978, 0.443957, 0.443901, 0.443881, 0.443827, 0.443881, 0.443836, 0.443862,
                                 0.443895, 0.443830, 0.443857, 0.443870, 0.443796, 0.443847, 0.443797, 0.443804, 0.443872, 0.443829] + [None]*22
    val_losses_100_student =    [0.448149, 0.446423, 0.445700, 0.445319, 0.445077, 0.444933, 0.444847, 0.444747, 0.444657, 0.444596, 0.444570, 0.444531, 0.444488, 0.444469, 0.444350, 0.444345, 0.444309, 0.444293,
                                 0.444229, 0.444233, 0.444218, 0.444216, 0.444189, 0.444147, 0.444206, 0.444193, 0.444143, 0.444106, 0.444109, 0.444136, 0.444148, 0.444061, 0.444089, 0.444091, 0.444076, 0.444094,
                                 0.444079, 0.444104, 0.444102, 0.444083, 0.444044, 0.444050, 0.444060, 0.444057, 0.444056, 0.444046, 0.444015, 0.444018, 0.444008, 0.443997]
    val_losses_100_small =      [0.448546, 0.446609, 0.445772, 0.445398, 0.445160, 0.444973, 0.444855, 0.444771, 0.444690, 0.444616, 0.444602, 0.444562, 0.444450, 0.444406, 0.444401, 0.444366, 0.444372, 0.444320,
                                 0.444278, 0.444265, 0.444227, 0.444315, 0.444237, 0.444226, 0.444211, 0.444121, 0.444148, 0.444128, 0.444184, 0.444167, 0.444163, 0.444197, 0.444202] + [None]*17

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_losses_200_student, label="(200,200,200) + KD", color="orange", marker='D', markevery=5)
    plt.plot(epochs, val_losses_200_small, label="(200,200,200)", linestyle="dashed", color="red", marker='D', markevery=5)
    plt.plot(epochs, val_losses_100_student, label="(100,100,100) + KD", color="green", marker='D', markevery=5)
    plt.plot(epochs, val_losses_100_small, label="(100,100,100)", linestyle="dashed", color="blue", marker='D', markevery=5)
    plt.xlabel("Epochs")
    plt.xticks(np.arange(1, 50, 5.0))
    plt.ylabel("Validation LogLoss")
    plt.legend()
    plt.savefig('../figures/kd_val_loss.png', dpi=500)
    plt.show()


def plot_latency_vs_auc_quantization():
    latency = [44.724, 8.524, 11.139, 14.047]
    auc = [0.8077, 0.8076, 0.8076, 0.8073]

    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.bar(np.arange(len(latency)), height=latency, color=['lawngreen', 'lightseagreen', 'royalblue', 'blueviolet'])
    plt.xticks(np.arange(len(latency)), ['Original', 'Dynamic', 'Static', 'QAT'])

    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('AUC', color=color)  # we already handled the x-label with ax1
    ax2.plot(auc, color=color, marker='D')
    ax2.set_yticks(np.arange(min(auc), max(auc)+0.0001, 0.0001))
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('figures/quant_latency.png', dpi=300)
    plt.show()


def plot_latency_vs_auc_qr():
    latency = [6.388, 44.724, 31.444, 25.028]
    auc = [0.8050, 0.8077, 0.8062, 0.8039]

    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.bar(np.arange(len(latency)), height=latency, color=['orange', 'lawngreen', 'lightseagreen', 'royalblue'])
    plt.xticks(np.arange(len(latency)), ['FwFM', 'DeepFwFM', 'QR (4)', 'QR (16)'])

    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('AUC', color=color)  # we already handled the x-label with ax1
    ax2.plot(auc, color=color, marker='D')
    ax2.set_yticks(np.arange(min(auc), max(auc)+0.001, 0.001))
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('../figures/qr_latency.png', dpi=300)
    plt.show()


def plot_latency_vs_auc():
    font = {'weight': 'bold',
            'size': 13}

    matplotlib.rc('font', **font)

    latency = [0.178, 2.018, 36.250, 142.776, 10080.782, 549.400, 7.062]  # 512
    auc = [0.7899, 0.7971, 0.8058, 0.8056, 0.8078, 0.8086, 0.8080]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color = 'black'
    #ax1.set_xlabel('Model')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.bar(np.arange(len(latency)), height=latency, color=['lightgreen', 'limegreen', 'darkgreen', 'lightblue', 'navy', 'royalblue', 'magenta'])
    plt.xticks(np.arange(len(latency)), ['LR', 'FM', 'FwFM', 'DeepFM', 'xDeepFM', 'DeepFwFM', 'xsDeepFwFM'])

    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('AUC', color=color)  # we already handled the x-label with ax1
    ax2.plot(auc, color=color, marker='D')
    #ax2.set_yticks(np.arange(0.7890, 0.8090, 0.005))
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('../figures/latency.png', dpi=500)
    plt.show()


def plot_latency_vs_batch_size():
    # idea from: https://chart-studio.plotly.com/~aman_cold/70/#/
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    batch_sizes = [64, 128, 256, 512]

    latency1 = [9.232, 14.276, 25.749, 49.082]
    latency2 = [10.139, 16.034, 27.146, 49.352]
    latency3 = [11.141, 16.404, 26.892, 46.676]
    # latency1 = [72.418, 143.632, 279.482, 549.400]

    throughput1 = np.array([b / l for (b, l) in zip(batch_sizes, latency1)]) * 1000
    throughput2 = np.array([b / l for (b, l) in zip(batch_sizes, latency2)]) * 1000
    throughput3 = np.array([b / l for (b, l) in zip(batch_sizes, latency3)]) * 1000
    #throughput4 = [b / l for (b, l) in zip(batch_sizes, latency4)]

    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('Batch Size (CPU)')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    for i, l1 in enumerate(latency1):
        ax1.bar(i - 0.2, l1, width=0.2, color='r', align='center')
        ax1.bar(i, latency2[i], width=0.2, color='g', align='center')
        ax1.bar(i + 0.2, latency3[i], width=0.2, color='b', align='center')
        #ax1.bar(i + 0.4, latency4[i], width=0.2, color='r', align='center')
    #plt.bar(np.arange(len(latency1)), height=latency1, color=plt.cm.get_cmap('winter')(rescale(batch_sizes)))
    plt.xticks(np.arange(len(batch_sizes)), map(str, batch_sizes))

    colors = {'Dynamic': 'r', 'Static': 'g', 'QAT': 'b'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'black'
    ax2.set_ylabel('Throughput (items/s)', color=color)  # we already handled the x-label with ax1
    ax2.plot([i - 0.2 for i in range(len(latency1))], throughput1, color='darkred', marker='D')
    ax2.plot([i for i in range(len(latency2))], throughput2, color='darkgreen', marker='D')
    ax2.plot([i + 0.2 for i in range(len(latency3))], throughput3, color='cornflowerblue', marker='D')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('../figures/batch_latency_quant.png', dpi=500)
    plt.show()


def plot_latency_vs_batch_size_qr(cpu=True):
    if cpu:
        batch_sizes = [64, 128, 256, 512]

        latency0 = [76.736, 151.032, 301.912, 609.880]
        latency1 = [67.244, 129.276, 253.910, 504.24]
        latency2 = [39.628, 75.712, 147.116, 288.920]
    else:
        batch_sizes = [512, 1024, 2048, 4096]

        latency0 = [6.772, 9.772, 14.440, 24.746]
        latency1 = [9.995, 10.356, 14.040, 20.046]
        latency2 = [9.275, 10.506, 14.431, 21.580]

    throughput0 = np.array([b / l for (b, l) in zip(batch_sizes, latency1)]) * 1000
    throughput1 = np.array([b / l for (b, l) in zip(batch_sizes, latency1)]) * 1000
    throughput2 = np.array([b / l for (b, l) in zip(batch_sizes, latency2)]) * 1000

    fig, ax1 = plt.subplots()

    color = 'black'
    if cpu:
        ax1.set_xlabel('Batch Size (CPU)')
    else:
        ax1.set_xlabel('Batch Size (CUDA)')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    for i, l1 in enumerate(latency1):
        ax1.bar(i - 0.2, latency0[i], width=0.2, color='r', align='center')
        ax1.bar(i , l1, width=0.2, color='g', align='center')
        ax1.bar(i + 0.2, latency2[i], width=0.2, color='b', align='center')
    plt.xticks(np.arange(len(batch_sizes)), map(str, batch_sizes))

    colors = {'QR 2': 'r', 'QR 4': 'g', 'QR 60': 'b'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')
    ax1.xaxis.grid(color='gray', linestyle='dashed')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'black'
    ax2.set_ylabel('Throughput (items/s)', color=color)  # we already handled the x-label with ax1
    ax2.plot([i - 0.2 for i in range(len(latency0))], throughput0, color='darkred', marker='D')
    ax2.plot([i for i in range(len(latency1))], throughput1, color='darkgreen', marker='D')
    ax2.plot([i + 0.2 for i in range(len(latency2))], throughput2, color='cornflowerblue', marker='D')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if cpu:
        fig.savefig('../figures/batch_latency_qr_cpu.png', dpi=500)
    else:
        fig.savefig('../figures/batch_latency_qr_gpu.png', dpi=500)
    plt.show()


def plot_profile():
    category_names = ['aten::mul', 'aten::sum',
                      'aten::as_strided', 'aten::select', 'aten::embedding', 'aten::addmm']
    results = {
        'FwFM': [30.33, 28.62, 7.01, 5.30, 5.26, 0],
        'DeepFwFM': [11.49, 10.03, 2.64, 1.83, 1.4, 59.91]
    }

    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        # ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    plt.savefig('figures/profile.png', dpi=300)
    plt.show()


#plot_kd_val_loss()
#plot_latency_vs_auc_quantization()
#plot_latency_vs_auc_qr()
plot_latency_vs_auc()
#plot_profile()
#plot_latency_vs_batch_size()
#plot_latency_vs_batch_size_qr(cpu=False)
