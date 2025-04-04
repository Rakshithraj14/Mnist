import numpy as np
from urllib import request
import gzip
import pickle

# File mapping: label → filename
mnist_files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}

# Download all MNIST files
def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in mnist_files.values():
        print(f"Downloading {name}...")
        request.urlretrieve(base_url + name, name)
    print("Download complete.\n")

# Load and save MNIST data into a pickle file
def save_mnist():
    data = {}
    
    # Load images (offset = 16 bytes for header)
    for key in ["train_images", "test_images"]:
        with gzip.open(mnist_files[key], 'rb') as f:
            data[key] = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28*28)

    # Load labels (offset = 8 bytes for header)
    for key in ["train_labels", "test_labels"]:
        with gzip.open(mnist_files[key], 'rb') as f:
            data[key] = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    # Save to mnist.pkl
    with open("mnist.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print("Data saved to mnist.pkl\n")

# Initialize: download and save dataset
def init_mnist():
    download_mnist()
    save_mnist()

# Load preprocessed MNIST dataset
def load_mnist():
    with open("mnist.pkl", "rb") as f:
        data = pickle.load(f)
    return data["train_images"], data["train_labels"], data["test_images"], data["test_labels"]

# Run only if script is executed directly
if __name__ == "__main__":
    init_mnist()
