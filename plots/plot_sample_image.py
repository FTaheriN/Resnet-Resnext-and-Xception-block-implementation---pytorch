import random
import matplotlib.pyplot as plt

def plot_sample_images(data):
  fig = plt.figure()

  for i in range(len(data)):
    idx = random.randint(0, len(data))
    img, label = data[idx]
    # print(img.shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(label)
    ax.axis('off')
    plt.imshow(img.permute(1,2,0))

    if i == 3:
        plt.show()
        break