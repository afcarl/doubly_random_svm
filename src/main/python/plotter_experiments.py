import matplotlib
import matplotlib.pyplot as plt
from training_comparison_functions import *

def plot_all():
    run_comparison_pred()
    run_comparison_expand()
    plot_large_scale()

def plot_large_scale():

    font = {'family': 'normal',
             'weight': 'normal',
             'size': 12}
    linewidth = 2
    figSize = (5,3)
    #
    #
    # ## speedup graph
    speedup = [21.132137031,20.240625,16.3974683544,9.0524109015,1]
    #
    cores = [40,30,20,10,1]
    #
    plt.ion()
    plt.figure(figsize=figSize)

    plt.plot(cores,speedup,'k-o',linewidth=linewidth)
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.rc('font',**font)
    plt.tight_layout()

    plt.savefig("speedup_58.pdf",)

    #
    # ## validation set graph
    #
    validation_error = [0.51,0.17,
    0.15,
    0.15,
    0.15,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.13,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.14,
    0.13,
    0.14,
    0.14,
    0.13,
    0.14,
    0.14,
    0.14,
    0.13,
    0.14,
    0.14,
    0.14,
    0.14,
    0.13,
    0.14,
    0.14,
    0.13,
    0.14,
    0.13,
    0.13,
    0.13,
    0.13,
    0.13,
    0.13,
    0.13,
    0.13,
    0.14,
    0.14,
    0.14,
    0.13,
    0.13]


    plt.figure(figsize=figSize)
    plt.plot(validation_error,"k",linewidth=linewidth)
    plt.xlim(0,50)
    plt.xlabel('Number of epochs')
    plt.ylabel('Validation Error')
    # plt.show()
    plt.rc('font',**font)
    plt.tight_layout()

    plt.savefig("covertype_validation.pdf",)


