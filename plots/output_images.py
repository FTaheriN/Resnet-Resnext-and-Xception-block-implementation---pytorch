import torch
import random
import matplotlib.pyplot as plt

def plot_output_images(data, res_pred, inc_pred, rnx_pred):
  classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  fig, ax = plt.subplots(3,3,figsize=(4,4));

  for i in range(len(data)):
    idx = random.randint(0, len(data))
    j = 0
    for model_pred in [res_pred, inc_pred, rnx_pred]:
       
        img, lab = data[idx]
        label = torch.argmax(model_pred[idx])
        
      # print(img.shape)

        ax[j,i].imshow(img.permute(1,2,0))

        ax[j,i].set_title(classes[label])
        ax[j,i].axis('off')
    #      plt.imshow(img.permute(1,2,0))
        j += 1
    if i == 2:
        plt.show()
        break