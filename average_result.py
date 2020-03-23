import numpy as np
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--filename', default="",type=str)
args = parser.parse_args()

#
# filelist = ["10fold1_1000",
filelist=[
    "1_1000",
    "2_1000",
"3_1000",
"4_1000",
"5_1000",
"6_1000",
"7_1000",
"8_1000",
"9_1000",
"10_1000"]


filelist = [args.filename + item for item in filelist]

filelist = ["exp/MESSIDOR/" + item for item in filelist]

whole_metric = np.zeros((10,13), dtype=float)

for i in range(len(filelist)):
    metrics= np.loadtxt(filelist[i] + "/result.txt", dtype=float)
    whole_metric[i] = metrics

mean = np.mean(whole_metric, axis=0)*100
std = np.std(whole_metric, axis=0)*100
print (whole_metric)
print (mean)
ci95 = 1.96 * std / np.sqrt(10 + 1)
# print (mean[:11]*100)

print ("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}"
       .format(mean[3], mean[0], mean[5], mean[7], mean[9], mean[4],
                       mean[1], mean[6], mean[8], mean[10], mean[2]))
print ("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}".format(
       std[3],std[0], std[5], std[7], std[9], std[4], std[1], std[6], std[8], std[10], std[2]))
print ("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}".format(
    ci95[3],ci95[0], ci95[5], ci95[7], ci95[9], ci95[4], ci95[1], ci95[6], ci95[8], ci95[10], ci95[2]))

print ("{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}"
       .format(mean[2], mean[3], mean[0], mean[4], mean[1]))
