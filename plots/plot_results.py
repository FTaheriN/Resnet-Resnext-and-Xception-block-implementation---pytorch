import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_loss():

    df = pd.read_csv("D:/Ms.C/DeepLearning/Homeworks/HW2/model/residual/Rep/Residual model_Rep.csv")
    train_rep1 = df.loc[df['mode'] == 'train'].reset_index().groupby('epoch').mean(numeric_only=True)
    testt_rep1 = df.loc[df['mode'] == 'val'].reset_index().groupby('epoch').mean(numeric_only=True)

    df = pd.read_csv("D:/Ms.C/DeepLearning/Homeworks/HW2/model/inception/Rep/Inception model_Rep.csv")
    train_rep2 = df.loc[df['mode'] == 'train'].reset_index().groupby('epoch').mean(numeric_only=True)
    testt_rep2 = df.loc[df['mode'] == 'val'].reset_index().groupby('epoch').mean(numeric_only=True)

    df = pd.read_csv("D:/Ms.C/DeepLearning/Homeworks/HW2/model/resnext/Rep/ResNeXt model_Rep.csv")
    train_rep3 = df.loc[df['mode'] == 'train'].reset_index().groupby('epoch').mean(numeric_only=True)
    testt_rep3 = df.loc[df['mode'] == 'val'].reset_index().groupby('epoch').mean(numeric_only=True)


    fig, ax = plt.subplots(2,3,figsize=(15,7));

    ax[0,0].plot(train_rep1['avg_train_top1_acc_till_current_batch'])
    ax[0,0].plot(testt_rep1['avg_val_top1_acc_till_current_batch'])
    ax[0,0].set_title('Residual Model');

    ax[0,1].plot(train_rep2['avg_train_top1_acc_till_current_batch'])
    ax[0,1].plot(testt_rep2['avg_val_top1_acc_till_current_batch'])
    ax[0,1].set_title('Inception Model');

    ax[0,2].plot(train_rep3['avg_train_top1_acc_till_current_batch'])
    ax[0,2].plot(testt_rep3['avg_val_top1_acc_till_current_batch'])
    ax[0,2].legend(['train accuracy', 'test accuracy'])
    ax[0,2].set_title('ResNeXt Model');

    ax[1,0].plot(train_rep1['loss_batch'], color="green")
    ax[1,1].plot(train_rep2['loss_batch'], color="green")
    ax[1,2].plot(train_rep3['loss_batch'], color="green")
    ax[1,2].legend(['train loss'])
    
    plt.show()
    return
