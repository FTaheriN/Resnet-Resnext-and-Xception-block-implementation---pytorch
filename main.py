# import os
# import glob 
import torch

from nets import *
from dataloaders import load_data
from utils import read_yaml_config
from deeplearning import train, test
from plots import plot_accuracy_loss, plot_output_images


############################## Reading Model Parameters ##############################
config = read_yaml_config()
random_seed = config['random_seed']
problem = config['problem']
DIR_TRAIN = config['dir_train'] 
DIR_TEST = config['dir_test'] 
learning_rate = config['learning_rate']
epochs = config['num_epochs']
batch_size = config['batch_size']
gamma = config['gamma']
step_size = config['step_size']
ckpt_save_freq = config['ckpt_save_freq']
model = config['model']
mode = config['mode']
ckpt_save_path = config['ckpt_save_path']
ckpt_name = config['ckpt_name']
loss = config['loss_func']

#################################### Loading Data ####################################


def main():
    train_dataset, test_dataset = load_data(DIR_TRAIN, DIR_TEST)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=2048,
                                                shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)
    
    device = "cpu"

    if model == "residual":
        MODEL = BaseModel(ResidualBlock)
    elif model == "inception":
        MODEL = BaseModel(InceptionBlock)
    else:
        MODEL = BaseModel(ResNeXtBlock)

    if mode == "train":
        print("Training model...")
        trainer = train(
            train_loader=train_loader,
            val_loader=test_loader,
            model = MODEL,
            model_name="Residual model",
            epochs=epochs,
            learning_rate=learning_rate,
            gamma = gamma,
            step_size = step_size,
            device=device,
            load_saved_model=False,
            ckpt_save_freq=ckpt_save_freq,
            ckpt_save_path=ckpt_save_path + model + "/",
            ckpt_path=ckpt_save_path + model + ckpt_name,
            report_path=ckpt_save_path + model + "/Rep/",
        )
    elif mode == "test":
        print("Testing model...")
        # test_accuracy = test(MODEL, ckpt_save_path + model + ckpt_name, train_loader, learning_rate)
        # print("Residual model accuracy on test data: ",sum(sum(test_accuracy, []))/len(test_accuracy))
        custom_model = BaseModel(ResidualBlock)
        model_path = "D:/Ms.C/DeepLearning/Homeworks/HW2/model/residual/ckpt_Residual model_epoch15.ckpt"
        residual_accuracy, residual_pred = test(custom_model, model_path, test_loader, 0.001)

        custom_model = BaseModel(InceptionBlock)
        model_path = "D:/Ms.C/DeepLearning/Homeworks/HW2/model/inception/ckpt_Inception model_epoch15.ckpt"
        inception_accuracy, inception_pred = test(custom_model, model_path, test_loader, 0.001)

        custom_model = BaseModel(ResNeXtBlock)
        model_path = "D:/Ms.C/DeepLearning/Homeworks/HW2/model/resnext/ckpt_ResNeXt model_epoch15.ckpt"
        resnext_accuracy, resnext_pred = test(custom_model, model_path, test_loader, 0.001)


        print("Residual model accuracy on test data: ",sum(sum(residual_accuracy, []))/len(residual_accuracy))
        print("Inception model accuracy on test data: ",sum(sum(inception_accuracy, []))/len(inception_accuracy))
        print("ResNeXt model accuracy on test data: ",sum(sum(resnext_accuracy, []))/len(resnext_accuracy))


        plot_output_images(test_dataset, residual_pred, inception_pred, resnext_pred)


    else:
        plot_accuracy_loss()


main()

