"""Module that wraps h5py for a simulation snapshot.
Allows loading by a directory and a snapshot number."""

import os.path
import re
import glob
import h5py
import numpy as np

def get_file(num, base, file_num=0):
    """Get a file descriptor from a simulation directory,
    snapshot number and optionally file number.
    Input:
        num - snapshot number
        base - simulation directory
        file_num - file number in the snapshot"""
    fname = base
    snap=str(num).rjust(3,'0')
    new_fname = os.path.join(base, "snapdir_"+snap)
    #Check for snapshot directory
    if os.path.exists(new_fname):
        fname = new_fname
    #Find a file
    fnames = glob.glob(os.path.join(fname, "snap_"+snap+"*hdf5"))
    f = h5py.File(fnames[0], 'r')
    return f

def get_all_files(num, base):
    """Gets a list of all files in this snapshot, by opening them in turn."""
    ff = get_file(num, base)
    files = [ff.filename,]
    ff.close()
    for i in range(1,3000):
        filename = re.sub(r"\.0\.hdf5","."+str(i)+".hdf5",files[0])
        #If we only have one file
        if filename == files[0]:
            break
        if os.path.exists(filename):
            files.append(filename)
        else:
            break
    return files


def get_baryon_array(name,num, base, file_num=0,dtype=np.float64):
    """Get a baryon array by name from a simulation directory.
    Input:
        name - Name of the array
        num - snapshot number
        base - simulation directory
        file_num - file number in the snapshot
        dtype - Type to give to array.
    Returns the named array formatter as a numpy double array."""
    f = get_file(num,base,file_num)
    bar=f["PartType0"]
    data=np.array(bar[name],dtype=dtype)
    f.close()
    return data

