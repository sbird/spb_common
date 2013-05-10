"""Module that wraps h5py for a simulation snapshot.
Allows loading by a directory and a snapshot number."""

import h5py
import numpy as np

def get_file(num, base, file_num=0):
    """Get a file descriptor from a simulation directory,
    snapshot number and optionally file number.
    Input:
        num - snapshot number
        base - simulation directory
        file_num - file number in the snapshot"""
    snap=str(num).rjust(3,'0')
    fname=base+"/snapdir_"+snap+"/snap_"+snap
    try:
        f=h5py.File(fname+"."+str(file_num)+".hdf5",'r')
    except IOError:
        if file_num == 0:
            fname=base+"/snap_"+snap
            f=h5py.File(fname+".hdf5",'r')
        else:
            raise IOError
    return f

def get_all_files(num, base):
    """Gets a list of all files in this snapshot, by opening them in turn."""
    ff = get_file(num, base)
    files = [ff.filename,]
    ff.close()
    for i in xrange(1,3000):
        try:
            ff = get_file(num,base,i)
            files.append(ff.filename)
            ff.close()
        except IOError:
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

