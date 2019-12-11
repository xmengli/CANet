import matplotlib.pyplot as plt
import csv


def draw_singleline(path):

    axis_x = []
    axis_y = []

    with open(path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            axis_x.append(int(row[1]))
            axis_y.append(float("%.5f" % float(row[2])))
    
    return axis_x, axis_y


def draw_averline(path):

    axis_y = []
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(csvreader):
            print (row)
            axis_y.append(float("%.5f" % float(row[0])))
    axis_x = list(range(0,len(axis_y)))
    return axis_x, axis_y


def baselines(limit,file_list, leg_list, flag_list, color,  name_x, name_y, name):



    for i in range(len(file_list)):
        if file_list[i].split("/")[-1][:4] in ["aver", "auc_", "loss"]:
            axis_preamd_x, axis_preamd_y = draw_averline(file_list[i])

        else:
            axis_preamd_x, axis_preamd_y = draw_singleline(file_list[i])

        plt.plot(axis_preamd_x[:limit], axis_preamd_y[:limit], flag_list[i], color = color[i])

    # plt.gca().legend((leg_list),loc='upper right', fontsize=8, bbox_to_anchor=(1.0, 0.9))
    plt.gca().legend((leg_list),loc='upper right', fontsize=8)

    # plt.gca().set_ylim([0,0.8])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(name_x, fontsize=15)
    plt.ylabel(name_y, fontsize=15)
    # plt.show()
    plt.savefig("figs/"+str(name)+".pdf",dpi=5000)


def plot_baseline_auc():

    file_list = [
        # "skin_baseline_100shot_imgnet/aver_all.csv",
        "skin_baseline_100shot/aver_all.csv",
        "skin_4conv_baseline_100shot/auc_aver_all.csv",
        "skin_baseline_5shot/aver_all.csv",
        "skin_4conv_baseline_5shot/auc_aver_all.csv",
        "skin_baseline_3shot/aver_all.csv",
    "skin_4conv_baseline_3shot/auc_aver_all.csv",
        "skin_baseline_1shot/aver_all.csv",
    "skin_4conv_baseline_1shot/auc_aver_all.csv"]
    #
    leg_list = [
        "DenseNet: k = 100",
        "4 ConvBlo: k = 100",
        "DenseNet: k = 5",
        "4 ConvBlo: k = 5",
        "DenseNet: k = 3",
        "4 ConvBlo: k = 3",
        "DenseNet: k = 1",
        "4 ConvBlo: k = 1"]

    # file_list = ["ours/our_5shot_aug_64_weigtask.csv", "ours/stand_5shot_aug_64.csv"]
    #
    # leg_list = ["ours, aug, difficulty", "ours, aug"]
    file_list = ["figs/"+item for item in file_list]
    flag_list = ["-","--","-","--", "-", "--", "-", "--"]
    color = ['green','green', 'blue','blue','red','red','grey','grey']
    baselines(limit, file_list, leg_list, flag_list, color, "Epoch", "AUC", "baseline_auc")

def plot_baseline_loss():


    file_list = [
        # "skin_baseline_100shot_imgnet/aver_all.csv",
        "skin_baseline_100shot/aver_all_loss.csv",
        "skin_4conv_baseline_100shot/loss_aver_all.csv",
        "skin_baseline_5shot/aver_all_loss.csv",
        "skin_4conv_baseline_5shot/loss_aver_all.csv",
        "skin_baseline_3shot/aver_all_loss.csv",
        "skin_4conv_baseline_3shot/loss_aver_all.csv",
        "skin_baseline_1shot/aver_all_loss.csv",
        "skin_4conv_baseline_1shot/loss_aver_all.csv"]
    #
    leg_list = [
        "DenseNet: k = 100",
        "4 ConvBlo: k = 100",
        "DenseNet: k = 5",
        "4 ConvBlo: k = 5",
        "DenseNet: k = 3",
        "4 ConvBlo: k = 3",
        "DenseNet: k = 1",
        "4 ConvBlo: k = 1"]

    # file_list = ["ours/our_5shot_aug_64_weigtask.csv", "ours/stand_5shot_aug_64.csv"]
    #
    # leg_list = ["ours, aug, difficulty", "ours, aug"]
    file_list = ["figs/" + item for item in file_list]
    flag_list = ["-", "--", "-", "--", "-", "--", "-", "--"]
    color = ['green', 'green', 'blue', 'blue', 'red', 'red', 'grey', 'grey']
    baselines(limit, file_list, leg_list, flag_list, color, "Epoch", "Loss", "baseline_loss")
    

def plot_ours_auc():


    file_list = ["ours/our_5shot_aug_64_weigtask.csv",

                 "ours/our_5shot_aug_64_weigtask_20.csv",
                 "ours/our_5shot_aug_64_weigtask_10.csv",
                 # "ours/stand_5shot_aug_64.csv",
                 "ours/stand_5shot_aug_32_600.csv",
                 "ours/stand_5shot_32.csv",
                 "ours/our_residual_32.csv",
                 "ours/ours_residual_64.csv",
                 "ours/our_224_input.csv"]

    leg_list = ["conv blocks,64f,84s+aug+DLf:1.5",
                "conv blocks,64f,84s+aug+DLf:2.0",
                "conv blocks,64f,84s+aug+DLf:1.0",
                # "ours:32f+aug+DLF:1.5",
                "conv blocks,64f,84s+aug", # 79.98
                "conv blocks,64f,84s",   # 79.34
                "resi   blocks,32f,84s", # 77.73
                "resi   blocks,64f,84s",  # 75.88
                "conv blocks,64f,224s"]  # 71.53
    file_list = ["figs/"+item for item in file_list]
    flag_list = ["--", "--", "--", "-","-","-","-","-"]
    baselines(limit, file_list, leg_list, flag_list, "Iterations", "AUC", "ours_auc")





if __name__ == '__main__':
    limit = 724

    x, y = draw_singleline("train_loss_idrid.csv")
    # x, y = draw_singleline(("train_loss.csv"))
    # x = x[:1100]
    # y = y[:1100]
    print (len(x))
    print (len(y))

    num = 5
    plt.plot(x[:limit:num], y[:limit:num])
    plt.gca().legend((["IDRiD dataset"]),loc='upper right', fontsize=15)
    # plt.plot(z[:limit:5], w[:limit:5])
    # plt.gca().legend((leg_list),loc='upper right', fontsize=8, bbox_to_anchor=(1.0, 0.9))
    # plt.gca().legend(,loc='upper right', fontsize=8)

    # plt.gca().set_ylim([0,0.8])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Interation", fontsize=15)
    plt.ylabel("Loss", fontsize=18)
    # plt.show()
    plt.savefig("figs_trainloss_idrid.pdf",dpi=5000)

    # x2, y2 = draw_singleline("val_acc.csv")
    # print (len(x2))
    # plt.plot(x2, y2)
    #
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # # plt.xlabel(, fontsize=15)
    # plt.ylabel("Joint Ac", fontsize=15)
    # # plt.show()
    # plt.savefig("figs_jointac.pdf",dpi=5000)
