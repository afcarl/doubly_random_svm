import matplotlib
import matplotlib.pyplot as plt

# # font = {'family': 'normal',
# #         'weight': 'bold',
# #         'size': 22}
#
#
# ## speedup graph
# # speedup = [21.132137031,20.240625,16.3974683544,9.0524109015,1]
#
# speedup = [4.8755656109,
# 5.4695431472,
# 5.8086253369,
# 5.8086253369,
# 5.8879781421,
# 6.2645348837,
# 6.4520958084,
# 6.8849840256,
# 7.4054982818,
# 7.534965035,
# 6.5902140673,
# 6.8849840256,
# 6.9741100324,
# 7.0195439739,
# 7.1122112211,
# 5.8879781421,
# 5.9366391185,
# 5.1678657074,
# 4.9769053118,
# 4.3273092369,
# 3.4424920128,
# 2.7628205128,
# 1.8771777003,
# 1]
#
# cores = [48,
# 46,
# 44,
# 42,
# 40,
# 38,
# 36,
# 34,
# 32,
# 30,
# 28,
# 26,
# 24,
# 22,
# 20,
# 18,
# 16,
# 14,
# 12,
# 10,
# 8,
# 6,
# 4,
# 2]
#
# # cores = [40,30,20,10,1]
#
# # plt.ion()
# # plt.figure(figsize=(20,12))
# colors = "brymcwg"
# leg = []
# # plt.plot(Ebatch.mean()*sp.ones(Eemp.shape[1]),color='black')
# ax = plt.scatter(cores,speedup)
# leg.append("Speedup")
# plt.legend(leg)
# plt.xlabel('Number of cores')
#
# plt.ylabel('Speedup')
# plt.axis([0, 49, 0, 10])
#
# # x_important = [1,20,30,40]
# x_important = range(0,49,2)
# plt.xticks(x_important, x_important)
#
#
# # font = {'size': 22}
#
# # matplotlib.rc('font', **font)
# #plt.ylim(0,1.3)
# # plt.axis('tight')
# plt.savefig("speedup_58.pdf",)
#
#
# ## validation set graph
#
# # validation_error = []



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


# plt.figure(figsize=(20,12))
# colors = "brymcwg"
leg = []
# # plt.plot(Ebatch.mean()*sp.ones(Eemp.shape[1]),color='black')
# ax = plt.scatter(cores,speedup)


plt.plot(validation_error)
leg.append("Validation Error")
plt.legend(leg)
plt.xlabel('Number of epochs')
plt.ylabel('Validation Error')
# plt.show()

plt.savefig("covertype_validation.pdf",)