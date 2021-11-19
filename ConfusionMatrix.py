from tensorflow.python.keras.models import Model, load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

classes = ['open', 'closed']#类别集合

def plot_confusion_matrix2(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def main():
    # 数据集实例化(创建数据集)
    x_val = np.load('dataset/x_val.npy').astype(np.float32)
    y_val = np.load('dataset/y_val.npy').astype(np.float32)
    model = load_model('models\\CNN+AM-97.32%.h5')
    y_pred = model.predict(x_val/255.)
    y_pred_logical = (y_pred > 0.5).astype(np.int)
    print(y_pred_logical)
    cm = confusion_matrix(y_val, y_pred_logical)
    # plot_confusion_matrix(cm, title='Normalized confusion matrix')
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]# 归一化
    plot_confusion_matrix2(cm, 'confusion_matrix.png', title='Normalized confusion matrix')



if __name__ == '__main__':
    main()