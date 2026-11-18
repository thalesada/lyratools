"""Appending analysis results to HDF5 files."""
import os

import h5py
import numpy as np


def add_to_file(data, label, h5file, mode="a"):
    """
    Write `data` under `label` in an open HDF5 file.
    mode 'o': overwrite; 'a': append values to one growing 1D array;
    'aa': append as a new row (all rows must have the same shape).
    """
    if mode == "o" and label in h5file:
        del h5file[label]
    if label not in h5file:
        h5file.create_dataset(label, data=[data], compression="gzip", chunks=True, maxshape=(None, *np.shape(data)))
    elif mode == "aa":
        h5file[label].resize(h5file[label].shape[0] + 1, axis=0)
        h5file[label][-1:] = data
    elif mode == "a":
        new = np.hstack([*h5file[label][:], data])
        del h5file[label]
        h5file.create_dataset(label, data=[new], compression="gzip", chunks=True, maxshape=(None, len(new)))
    elif mode != "o":
        raise ValueError(f"invalid mode '{mode}'")


def write_to_file(data, label, h5file, mode="a"):
    """Write one label, or a list of labels with matching data, to an open HDF5 file."""
    if isinstance(label, str):
        add_to_file(data, label, h5file, mode)
    else:
        for d, l in zip(data, label):
            add_to_file(d, l, h5file, mode)


def add_to_storage(data, label, path, overwrite=False):
    """Open (creating directories), write/append `data` under `label`, close."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "a") as f:
        if label not in f:
            f.create_dataset(label, data=[data], compression="gzip", chunks=True, maxshape=(None, *np.shape(data)))
        elif overwrite:
            f[label][...] = data
        else:
            f[label].resize(f[label].shape[0] + 1, axis=0)
            f[label][-1:] = data


def storage_path(simpath, snap, name="MyDataset"):
    """Conventional location: <sim>/postprocessing/<name>_<snap>.hdf5"""
    return os.path.join(simpath, "postprocessing", f"{name}_{snap:03d}.hdf5")


def add_dataset(h5file, label):
    if label not in h5file:
        h5file.create_dataset(label, dtype="f")
    return h5file[label]
