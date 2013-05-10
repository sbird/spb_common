# -*- coding: utf-8 -*-
"""A small module for computing the smoothing length of a particle simulation.
(Non-trivial in Arepo)"""

import math
import numpy as np

def get_smooth_length(bar):
    """Figures out if the particles are from AREPO or GADGET
    and computes the smoothing length.
    Note the Volume array in HDF5 is comoving and this returns a comoving smoothing length
    If we are Arepo, this smoothing length is  cell radius, where
    cell volume = 4/3 π (cell radius) **3 and cell volume = mass / density
    Arguments:
        Baryon particles from a simulation
    Returns:
        Array of smoothing lengths in code units.
    """
    #Are we arepo? If we are a modern version we should have this array.
    if np.any(np.array(bar.keys()) == 'Volume'):
        volume=np.array(bar["Volume"])
        radius = (3*volume/4/math.pi)**(0.33333333)
    elif np.any(np.array(bar.keys()) == 'Number of faces of cell'):
        rho=np.array(bar["Density"])
        mass=np.array(bar["Masses"])
        volume = mass/rho
        radius = (3*volume/4/math.pi)**(0.33333333)
    else:
        #If we are gadget, the SmoothingLength array is actually the smoothing length.
        radius=np.array(bar["SmoothingLength"])
    return radius
