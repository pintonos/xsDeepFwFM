import matplotlib.pyplot as plt
import numpy as np


def plot_kd_val_loss():
    epochs = np.arange(1, 50)
    val_losses_200_student = [] # TODO
    val_losses_200_small = [] # TODO
    val_losses_100_student = [] # TODO
    val_losses_100_small = [0.448422, 0.446495, 0.445742, 0.445319, 0.445052, 0.444879, 0.444743, 0.444638, 0.444595, 0.444482, 0.444461, 0.444438, 0.444387, 0.444376, 0.444338, 0.444296, 0.444263, 0.444256, 0.444247, 0.444249, 0.444247, 0.444244, 0.444244, 0.444268, 0.444233, 0.444266, 0.444213, 0.444234, 0.444230, 0.444246, 0.444190, 0.444172, 0.444201, 0.444166, 0.444208, 0.444131, 0.444099, 0.444113, 0.444167, 0.444097, 0.444105, 0.444145]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_losses_200_student, label="(200,200,200) + KD", color="orange", marker='D')
    plt.plot(epochs, val_losses_200_small, label="(200,200,200)", linestyle="dashed", color="orange", marker='D')
    plt.plot(epochs, val_losses_100_student, label="(100,100,100) + KD", color="blue", marker='D')
    plt.plot(epochs, val_losses_100_small, label="(100,100,100)", linestyle="dashed", color="blue", marker='D')
    plt.xlabel("Epochs")
    plt.xticks(np.arange(1, 50, 1.0))
    plt.ylabel("Validation LogLoss")
    plt.legend()
    plt.savefig('figures/kd_val_loss.png', dpi=300)
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
    fig.savefig('figures/qr_latency.png', dpi=300)
    plt.show()


def plot_latency_vs_auc():
    latency = [41.17, 331.48, 217.24, 178.26, 54.62, 48.10, 61.67]  # 512
    auc = [0.8050, 0.8077, 0.8062, 0.8039, 0.8076, 0.8076, 0.8073]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'black'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Latency (ms)', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.bar(np.arange(len(latency)), height=latency, color=['orange', 'green', 'navy', 'royalblue', 'indigo', 'darkviolet', 'mediumorchid'])
    plt.xticks(np.arange(len(latency)), ['FwFM', 'DeepFwFM', 'QR (4)', 'QR (16)', 'Dynamic', 'Static', 'QAT'])

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
    fig.savefig('figures/latency.png', dpi=300)
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


plot_kd_val_loss()
#plot_latency_vs_auc_quantization()
#plot_latency_vs_auc_qr()
#plot_latency_vs_auc()
#plot_profile()
