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
    limit = 100
    # plot_baseline_auc()
    # plot_baseline_loss()
    # plot_ours_auc()
    # file_list = ["pm_full_valacc_preamd.csv",
    #              "pm_full_valacc_preimgnet.csv",
    #              "pm_5shot_valacc.csv",
    #              "pm_5shot_standaug.csv",
    #              "pm_1shot_valacc.csv",
    #              "pm_1shot_standaug.csv"]

    # file_list = ["loss_pm_full_preamd.csv",
    #              "loss_pm_full_preimgnet.csv",
    #              "loss_pm_5shot.csv",
    #              "loss_pm_5shot_standaug.csv",
    #              "loss_pm_1shot.csv",
    #              "loss_pm_1shot_standaug.csv"]


    # file_list = ["amd_valacc/amd_valacc_pmpre.csv",
    #              "amd_valacc/amd_valacc_imgnetpre.csv",
    #              "amd_valacc/amd_valacc_10shot.csv",
    #             "amd_valacc/amd_valacc_5shot.csv"]
    # leg_list = ['full data (stand aug, AMD pretrain)',
    #             'full data (stand aug, ImageNet pretrain)',
    #             '5 shot (extreme aug, AMD pretrain)',
    #             '5 shot (stand aug, AMD pretrain)',
    #             '1 shot (extreme aug, AMD pretrain)',
    #             '1 shot (stand aug, AMD pretrain)']
    #
    # leg_list = ['full data (stand aug, AMD pretrain)',
    #             'full data (stand aug, ImageNet pretrain)',
    #             '5 shot (extreme aug, AMD pretrain)',
    #             '5 shot (stand aug, AMD pretrain)',
    #             '1 shot (extreme aug, AMD pretrain)',
    #             "1 shot (syand aug, AMD pretrain)"]

    # leg_list = ["full data (stand aug, PM pretrain)",
    #             "full data (stand aug, ImageNet pretrain)",
    #             "10 shot (stand aug, PM pretrain)",
    #             "5 shot (stand aug, PM pretrain)"]


    # file_list = ["skin_baseline_5shot/aver_df_ak.csv",
    #              "skin_baseline_5shot/aver_df_vasc.csv",
    #              "skin_baseline_5shot/aver_vasc_ak.csv",
    #              "skin_baseline_5shot/aver_all.csv"]

    # file_list = ["skin_baseline_10shot/aver_df_ak.csv",
    #              "skin_baseline_10shot/aver_df_vasc.csv",
    #              "skin_baseline_10shot/aver_vasc_ak.csv",
    #              "skin_baseline_10shot/aver_all.csv"]


    from PIL import Image

    img = Image.open("/home/xmli/medical/MESSIDOR/Base11/20051019_38557_0100_PP.tif")
    img = img.resize((350, 350), Image.ANTIALIAS)

    img.save("image.jpg")
    # cropp_img = img[175-112:175+112, 175-112:175+112,:]
    crop_img = img.crop((175-112, 175-112, 175+112, 175+112))
    crop_img.save("crop_image.jpg")