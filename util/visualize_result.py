
import matplotlib.pyplot as plt
from skimage import io
result_file = '../log/manhattan_existence.txt'


with open(result_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        image_path = line.split(',')[0]
        label = line.split(',')[-1]
        img = io.imread(image_path)
        plt.imshow(img)
        plt.title('label: {}'.format(label))
        plt.show()
