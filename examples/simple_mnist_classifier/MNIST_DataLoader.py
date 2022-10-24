import hashlib
import os
from urllib.request import urlretrieve
import gzip
import shutil
import codecs
import numpy as np
import sys


SN3_PASCALVINCENT_TYPEMAP = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.float64,
}


download_url = "http://yann.lecun.com/exdb/mnist/"
resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
]


def get_MNIST_dataset(root):
    download_dataset(download_url, resources, root)
    extract_dataset(root, resources)
    train_data, train_labels = load_data(True, root)
    test_data, test_labels = load_data(False, root)
    return train_data, train_labels, test_data, test_labels


def download_dataset(download_url, resources, root):
    for filename, md5 in resources:
        filepath = os.path.join(root, filename)
        download_path = download_url + filename
        if not os.path.exists(filepath):
            urlretrieve(download_path, filepath)

        assert md5 == hashlib.md5(open(filepath, "rb").read()).hexdigest()


def extract_dataset(root, resources):
    for filename, _ in resources:
        filepath = os.path.join(root, filename)
        with gzip.open(filepath, "rb") as f_in:
            with open(filepath[:-3], "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def read_sn3_pascalvincent_ndarray(path: str):
    with open(path, "rb") as f:
        data = f.read()
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    np_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    num_bytes_per_value = np.iinfo(np_type).bits // 8
    needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
    parsed = np.frombuffer(bytearray(data), dtype=np_type, offset=(4 * (nd + 1)))
    if needs_byte_reversal:
        parsed = parsed.flip(0)

    return parsed.reshape(*s)


def load_data(train, root):
    image_file = f"{'train' if train else 't10k'}-images-idx3-ubyte"
    data = read_sn3_pascalvincent_ndarray(os.path.join(root, image_file))

    label_file = f"{'train' if train else 't10k'}-labels-idx1-ubyte"
    targets = read_sn3_pascalvincent_ndarray(os.path.join(root, label_file))

    return data, targets
