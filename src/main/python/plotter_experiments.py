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
    speedup_old = [21.132137031,20.240625,16.3974683544,9.0524109015,1]
    #
    cores_old = [40,30,20,10,1]

    speedup = [ 9.056561086,10.1598984772,10.7897574124,10.7897574124,10.9371584699,
                11.636627907,11.9850299401,12.7891373802,13.7560137457,13.9965034965,
                12.2415902141,12.7891373802,12.9546925566,13.0390879479,13.2112211221,
                10.9371584699,11.0275482094,9.5995203837,9.2448036952,8.0381526104,
                6.3945686901,5.1320512821,3.4869337979,1.8575406032,1.0]

    cores = [48,46,44,42,40,
             38,36,34,32,30,
             28,26,24,22,20,
             18,16,14,12,10,
             8,6,4,2,1]
    #
    plt.ion()
    plt.figure(figsize=figSize)
    plt.axis([0, 49, 0, 15])
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
