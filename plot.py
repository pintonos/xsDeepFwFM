import matplotlib.pyplot as plt
import numpy as np


def plot_kd_val_loss():
    epochs = np.arange(1, 6)
    val_losses_128_student = [0.451030, 0.449960, 0.449685, 0.449376, 0.449314]
    val_losses_128_small = [0.447326, 0.445907, 0.445444, 0.445095, 0.444981]
    val_losses_64_student = [0.451616,0.450325, 0.449866, 0.449531, 0.449425]
    val_losses_64_small = [0.447748, 0.446368, 0.445790, 0.445482, 0.445273]
    val_losses_32_student = [0.451784, 0.450430, 0.450050, 0.449770, 0.449682]
    val_losses_32_small = [0.447997, 0.446669, 0.446322, 0.446094, 0.445911]
    val_losses_16_student = [0.452229, 0.451208, 0.450208, 0.450020, 0.449928]
    val_losses_16_small = [0.448191, 0.447019, 0.446597, 0.446390, 0.446236]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_losses_128_student, label="(128,128,128) + KD", color="orange", marker='D')
    plt.plot(epochs, val_losses_128_small, label="(128,128,128)", linestyle="dashed", color="orange", marker='D')
    plt.plot(epochs, val_losses_64_student, label="(64,64,64) + KD", color="yellow", marker='D')
    plt.plot(epochs, val_losses_64_small, label="(64,64,64)", linestyle="dashed", color="yellow", marker='D')
    plt.plot(epochs, val_losses_32_student, label="(32,32,32) + KD", color="blue", marker='D')
    plt.plot(epochs, val_losses_32_small, label="(32,32,32)", linestyle="dashed", color="blue", marker='D')
    plt.plot(epochs, val_losses_16_student, label="(16,16) + KD", color="red", marker='D')
    plt.plot(epochs, val_losses_16_small, label="(16,16)", linestyle="dashed", color="red", marker='D')
    plt.xlabel("Epochs")
    plt.xticks(np.arange(1, 6, 1.0))
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
    latency = [6.388, 44.724, 31.444, 25.028, 8.524, 11.139, 14.047]
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
