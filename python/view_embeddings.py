"""Script to show a PCA of triplet network embeddings.

The network embeddings are read from a formatted text file.

"""
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def read_array(filename):
    """Read an array from a formatted text file.

    Args:
        filename (str): Path of the file to read.

    Returns:
        numpy.array.

    """
    with open(filename, mode='r') as f:
        # On the first 3 lines, read the datatype, the number of dimensions,
        # and the sizes along each dimension.
        datatype = f.readline().strip()
        num_dimensions = int(f.readline().strip())
        shape_str = f.readline().strip().split(' ')
        shape = [int(x) for x in shape_str]
        array_length = np.prod(shape)
        
        # Read the data into a vector
        array = np.fromstring(f.readline(), dtype=float, sep=' ')
        
    # Return a numpy array with the prescribed shape
    assert array.shape[0] == array_length
    array = array.reshape(shape)
    return array

# Load the data
work_path = r'<PATH TO SET>'
embeddings = read_array(os.path.join(work_path, 'test_images_embeddings.txt'))
labels = read_array(os.path.join(work_path, 'test_labels.txt'))

# Calculate the PCA
pca = PCA(n_components=2)
low_dim_embeddings = pca.fit_transform(embeddings)

# Plot points by class
colors = np.array([
    [0,168,229],
    [255,115,13],
    [72,206,255]], dtype=float) / 255.
fig = plt.figure(figsize=(8., 6.))
ax = fig.add_subplot(111)
handles = []
for label in range(3):
    selection = (labels == label)
    handles.append(ax.scatter(
        low_dim_embeddings[selection, 0], low_dim_embeddings[selection, 1],
        c=colors[label:label+1, :], linewidths=0.2))
ax.set_title('Projection of the embeddings on the first two PCA components')
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.legend(
    handles, ['Label {}'.format(label) for label in range(3)],
    scatterpoints=1, markerscale=1.2)
fig.savefig(os.path.join(work_path, 'pca.png'), dpi=144)
