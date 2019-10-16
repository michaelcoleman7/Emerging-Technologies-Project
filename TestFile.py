import gzip
import matplotlib.pyplot as plt
import numpy as np

with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    file_content = f.read()
file_content[0:4]

l = file_content[16:800]
image = ~np.array(list(file_content[16:800])).reshape(28,28).astype(np.uint8)
plt.imshow(image, cmap='gray')
plt.show()