# -*- coding: utf-8 -*-
"""Split off module to load halo catalogues and export a list of mass and positions"""

import numpy as np
import readsubfHDF5
import readsubf

#Internal gadget mass unit: 1e10 M_sun/h in g/h
UnitMass_in_g=1.989e43
#1 M_sun in g
SolarMass_in_g=1.989e33


def is_masked(halo,sub_mass,sub_cofm, sub_radii):
    """Find out whether a halo is a mere satellite and if so mask it"""
    near=np.where(np.all((np.abs(sub_cofm[:,:]-sub_cofm[halo,:]) < sub_radii[halo]),axis=1))
    #If there is a larger halo nearby, mask this halo
    return np.size(np.where(sub_mass[near] > sub_mass[halo])) == 0

def find_wanted_halos(num, base, min_mass, dist=1):
    """When handed a halo catalogue, remove from it the halos that are within dist virial radii of other, larger halos.
    Select halos via their M_200 mass, defined in terms of the critical density.
    Arguments:
        num - snapnumber
        base - simulation directory
        min_mass - minimum mass of halos to use
    Returns:
        ind - list of halo indices used
        sub_mass - halo masses in M_sun /h
        sub_cofm - halo positions
        sub_radii - dist*R_Crit200 for halo radii"""
    try:
        subs=readsubf.subfind_catalog(base,num,masstab=True,long_ids=True)
        #Get list of halos resolved, using a mass cut; cuts off at about 2e9 for 512**3 particles.
        ind=np.where(subs.group_m_crit200 > min_mass)
        #Store the indices of the halos we are using
        #Get particle center of mass, use group catalogue.
        sub_cofm=np.array(subs.group_pos[ind])
        #halo masses in M_sun/h: use M_200
        sub_mass=np.array(subs.group_m_crit200[ind])*UnitMass_in_g/SolarMass_in_g
        #r200 in kpc.
        sub_radii = np.array(subs.group_r_crit200[ind])
        del subs
    except IOError:
        # We might have the halo catalog stored in the new format, which is HDF5.
        subs=readsubfHDF5.subfind_catalog(base, num,long_ids=True)
        #Get list of halos resolved, using a mass cut; cuts off at about 2e9 for 512**3 particles.
        ind=np.where(subs.Group_M_Crit200 > min_mass)
        #Store the indices of the halos we are using
        #Get particle center of mass, use group catalogue.
        sub_cofm=np.array(subs.GroupPos[ind])
        #halo masses in M_sun/h: use M_200
        sub_mass=np.array(subs.Group_M_Crit200[ind])*UnitMass_in_g/SolarMass_in_g
        #r200 in kpc/h (comoving).
        sub_radii = np.array(subs.Group_R_Crit200[ind])
        del subs

    sub_radii*=dist
    #For each halo
    ind2=np.where([is_masked(ii,sub_mass,sub_cofm,sub_radii) for ii in xrange(0,np.size(sub_mass))])
    ind=(np.ravel(ind)[ind2],)
    sub_mass=sub_mass[ind2]
    sub_cofm=sub_cofm[ind2]
    sub_radii=sub_radii[ind2]
    return (ind, sub_mass,sub_cofm,sub_radii)

