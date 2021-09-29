import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

def xiaoshu(s):
    xiaoshu_new=str(s)

    if xiaoshu_new.count(".") ==1:
       left,right = xiaoshu_new.split(".")
       if left.isdigit() and right.isdigit():
           print(1111)
           return True

       elif left.startswith('-') and left.count('-') == 1 and right.isdigit():
            lleft = left.split('-')[-1]
            if lleft.isdigit():
                return True

    return False

def plot_acc_loss(loss, acc, epoch):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()  # 共享x轴

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("test_loss")
    par1.set_ylabel("test-accuracy")

    # plot curves
    p1, = host.plot(range(len(loss)), loss, label="loss")
    p2, = par1.plot(range(len(acc)), acc, label="accuracy")

    # set location of the legend,
    # 1->rightup corner, 2->leftup corner, 3->leftdown corner
    # 4->rightdown corner, 5->rightmid ...
    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    # set the range of x axis of host and y axis of par1
    # host.set_xlim([-200, 5200])
    # par1.set_ylim([-0.1, 1.1])

    plt.draw()

    plt.savefig("acc2-"+str(epoch)+".jpg")
    plt.show()

eval_loss_list = []
eval_acctrain_list=[]
with open('train.txt', encoding='utf-8') as file:
    line_list = [k.strip() for k in file.readlines()]  # 用 strip()移除末尾的空格
i=1
for txt_str in line_list:

    if "137/137" in txt_str:
        print(txt_str)
        print(float(txt_str.split('val_loss:')[1].split(' -')[0]))
        print(float((txt_str.split('val_loss:')[1].split(' -')[1]).split('val_acc:')[1]))
        # print(float(txt_str.split('val_loss:')[1].split('val_acc:')[1].split(' - val_')[0]))
        eval_loss_list.append(float(txt_str.split('val_loss:')[1].split(' -')[0]))
        eval_acctrain_list.append(float((txt_str.split('val_loss:')[1].split(' -')[1]).split('val_acc:')[1]))
        # eval_acctrain_list.append(float(txt_str.split('loss:')[1].split('acc:')[1].split(' - val_')[0]))
        # if xiaoshu(txt_str.split('the loss_rate is :  ')[-1]):
        #     eval_loss_list.append(float(txt_str.split('the loss_rate is :  ')[-1]))
    i+=1
#     if xiaoshu(txt_str.split('acc_val is :  ')[-1]):
#
#         print(txt_str.split('acc_val is :  ')[-1])
#         eval_acctrain_list.append(float(txt_str.split('acc_val is :  ')[-1])+0.06)
#
# print(eval_acctrain_list)
#
#
epoch = 50
plot_acc_loss(eval_loss_list, eval_acctrain_list, epoch)