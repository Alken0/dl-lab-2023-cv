import numpy as np
import matplotlib.pyplot as plt


def main():
    titles = [
        "linear",
        "convolutional",
        "transformer",
        "transformer-shared-qk"
    ]
    paths = [
        "results/segmentation/linear/2023_05_11_14_23_50",
        "results/segmentation/convolutional/2023_05_11_15_59_44",
        "results/segmentation/transformer/2023_05_11_19_52_45",
        "results/segmentation/transformer/2023_05_11_19_01_04",
    ]
    for title, path in zip(titles, paths):
        with open(f'{path}/imou.npy', 'rb') as f:
            ious = np.load(f)
            classes = np.load(f)
            mean = np.mean(ious, axis=1)
            print(mean[-1])
        plt.clf()

        for i in range(len(classes)):
            plt.plot(ious[:, i], label=classes[i])

        plt.xlabel("Epochs")
        plt.ylabel("IoU")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # order important
        plt.tight_layout()

        plt.savefig(f'{path}/plot.png')


if __name__ == "__main__":
    main()
