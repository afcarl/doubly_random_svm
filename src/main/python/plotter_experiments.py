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
    # ## after each epoche
    # validation_error = [0.51,
    # 0.17,
    # 0.15,
    # 0.15,
    # 0.15,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.14,
    # 0.13,
    # 0.13,
    # 0.13,
    # 0.13,
    # 0.13,
    # 0.13,
    # 0.13,
    # 0.13,
    # 0.14,
    # 0.14,
    # 0.14,
    # 0.13,
    # 0.13]


    ## validation error
    ## after each input batch evaluated on the whole kernel
    validation_error = [0.491087344029,
    0.286096256684,
    0.24688057041,
    0.298573975045,
    0.24688057041,
    0.226381461676,
    0.222816399287,
    0.210338680927,
    0.22192513369,
    0.206773618538,
    0.210338680927,
    0.19696969697,
    0.196078431373,
    0.193404634581,
    0.205882352941,
    0.194295900178,
    0.187165775401,
    0.19073083779,
    0.191622103387,
    0.197860962567,
    0.200534759358,
    0.192513368984,
    0.195187165775,
    0.199643493761,
    0.188948306595,
    0.19073083779,
    0.183600713012,
    0.188948306595,
    0.18449197861,
    0.183600713012,
    0.186274509804,
    0.179144385027,
    0.180035650624,
    0.187165775401,
    0.186274509804,
    0.185383244207,
    0.176470588235,
    0.19073083779,
    0.187165775401,
    0.180926916221,
    0.170231729055,
    0.18449197861,
    0.169340463458,
    0.188057040998,
    0.173796791444,
    0.179144385027,
    0.17201426025,
    0.176470588235,
    0.17825311943,
    0.181818181818,
    0.186274509804,
    0.166666666667,
    0.177361853832,
    0.167557932264,
    0.180926916221,
    0.171122994652,
    0.16577540107,
    0.17201426025,
    0.167557932264,
    0.167557932264,
    0.168449197861,
    0.167557932264,
    0.166666666667,
    0.174688057041,
    0.168449197861,
    0.162210338681,
    0.170231729055,
    0.16577540107,
    0.163992869875,
    0.167557932264,
    0.163992869875,
    0.161319073084,
    0.16577540107,
    0.160427807487,
    0.164884135472,
    0.159536541889,
    0.166666666667,
    0.171122994652,
    0.156862745098,
    0.169340463458,
    0.156862745098,
    0.163101604278,
    0.154188948307,
    0.174688057041,
    0.157754010695,
    0.166666666667,
    0.155971479501,
    0.172905525847,
    0.155080213904,
    0.166666666667,
    0.154188948307,
    0.168449197861,
    0.155971479501,
    0.169340463458,
    0.156862745098,
    0.176470588235,
    0.150623885918,
    0.180926916221,
    0.159536541889,
    0.168449197861,
    0.152406417112,
    0.171122994652,
    0.159536541889,
    0.167557932264,
    0.153297682709,
    0.172905525847,
    0.155971479501,
    0.173796791444,
    0.158645276292,
    0.168449197861,
    0.157754010695,
    0.162210338681,
    0.152406417112,
    0.160427807487,
    0.148841354724,
    0.157754010695,
    0.147058823529,
    0.150623885918,
    0.156862745098,
    0.153297682709,
    0.147950089127,
    0.151515151515,
    0.160427807487,
    0.151515151515,
    0.155971479501,
    0.151515151515,
    0.157754010695,
    0.150623885918,
    0.151515151515,
    0.151515151515,
    0.153297682709,
    0.153297682709,
    0.155080213904,
    0.145276292335,
    0.155971479501,
    0.151515151515,
    0.155971479501,
    0.152406417112,
    0.155080213904,
    0.153297682709,
    0.146167557932,
    0.156862745098,
    0.149732620321,
    0.156862745098,
    0.147950089127,
    0.159536541889,
    0.141711229947,
    0.155080213904,
    0.150623885918,
    0.156862745098,
    0.152406417112,
    0.161319073084,
    0.151515151515,
    0.163101604278,
    0.152406417112,
    0.156862745098,
    0.152406417112,
    0.159536541889,
    0.149732620321,
    0.155971479501,
    0.145276292335,
    0.154188948307,
    0.151515151515,
    0.155080213904,
    0.145276292335,
    0.154188948307,
    0.145276292335,
    0.154188948307]

    plt.figure(figsize=figSize)
    plt.plot(validation_error,"k",linewidth=linewidth)
    # plt.xlim(0,50)
    # plt.xlabel('Number of epochs')
    plt.xlabel('Number of predicate batches')
    plt.ylabel('Validation Error')
    # plt.show()
    plt.rc('font',**font)
    plt.tight_layout()

    plt.savefig("covertype_validation.pdf",)

plot_large_scale()