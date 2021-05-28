import pandas as pd
import matplotlib.pyplot as plt


def feature_plotter(data):
    features = pd.read_csv(data)

    dependent = features.pop(features.columns[-1])

    fontsize = 12

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.scatter(features['Population (discrete data)'], dependent, s=5)
    ax1.set_xlabel('Population', fontsize=fontsize)
    ax1.set_ylabel('Cases', fontsize=fontsize)
    ax1.set_title('A', fontsize=fontsize)

    ax2.scatter(features['Tests (discrete data)'], dependent, s=5)
    ax2.set_xlabel('Tests', fontsize=fontsize)
    #ax2.set_ylabel('Cases', fontsize=fontsize)
    ax2.set_title('B', fontsize=fontsize)

    ax3.scatter(features['Gini - gov 2019 (continuous data)'], dependent, s=5)
    ax3.set_xlabel('Gini', fontsize=fontsize)
    ax3.set_xticks([0.4, 0.45, 0.5, 0.55])
    ax3.set_ylabel('Cases', fontsize=fontsize)
    ax3.set_title('C', fontsize=fontsize)

    ax4.scatter(features['% urban population (continuous data)'], dependent, s=5)
    ax4.set_xlabel('% urban population', fontsize=fontsize)
    #ax4.set_ylabel('Cases', fontsize=fontsize)
    ax4.set_title('D', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    feature_plotter('Final US Data.csv')
