"""Module to test the cold_gas code"""

import matplotlib
matplotlib.use('PDF')

import cold_gas
import myname
import numpy as np
import hdfsim
import random
import matplotlib.pyplot as plt
import save_figure

def setup_test(molec,sim):
    name = myname.get_name(sim)

    f=hdfsim.get_file(3,name,0)

    redshift=f["Header"].attrs["Redshift"]
    hubble=f["Header"].attrs["HubbleParam"]
    bar = f["PartType0"]

    cold = cold_gas.RahmatiRT(redshift, hubble, molec=molec)

    return (cold, bar)

def plot_vs_den(nH, thing):
    """Plot a sample of a thing vs density"""
    ind = np.where(nH < 1e-3)
    ints = random.sample(xrange(np.size(ind)),500)
    plt.semilogx(nH[ind][ints], thing[ind][ints], 'o')

    ind = np.where(nH >= 1e-3)
    ints = random.sample(xrange(np.size(ind)),3000)
    plt.semilogx(nH[ind][ints], thing[ind][ints], 'o')

    for thresh in (0.05, 0.15):
        ind = np.where(nH >= thresh)
        if np.size(ind) > 0:
            if np.size(ind) < 5000:
                plt.semilogx(nH[ind], thing[ind], 'o')
            else:
                ints = random.sample(xrange(np.size(ind)),5000)
                plt.semilogx(nH[ind][ints], thing[ind][ints], 'o')

def plot_neut_sim(molec, sim):
    """Plot neutral fraction for a particular sim"""
    (cold, bar)= setup_test(molec, sim)
    nHI = cold.get_reproc_HI(bar)

    nH=cold.get_code_rhoH(bar)
    plot_vs_den(nH, nHI)

def plot_temp_sim(molec, sim):
    """Plot temperature for a particular sim"""
    (cold, bar)= setup_test(molec, sim)
    nH=cold.get_code_rhoH(bar)
    temp = cold.get_temp(nH, bar)
    plot_vs_den(nH, temp)

def plot_test():
    """Test which plots the neutral fraction for a bunch of particles"""

    # plot_file(True,5)
    # save_figure.save_figure("test_cold_gas5_molec")
    # plt.clf()
    #
    # plot_file(False,5)
    # save_figure.save_figure("test_cold_gas5_no_molec")
    # plt.clf()

    plot_neut_sim(False,1)
    plt.semilogx(np.ones(3)*0.168, (0.0,0.5,1),"-")
#     plt.xticks((0.1,0.15,0.2,0.25))
#     plt.xlim(0.1,0.3)
    save_figure.save_figure("test_cold_gas1_no_molec")
    plt.clf()

#     plot_temp_sim(False,1)
#     plt.loglog(np.ones(2)*0.168, (1e3,5e7),"-")
#     plt.xticks((0.1,0.15,0.2,0.25))
#     plt.xlim(0.1,0.3)
#     save_figure.save_figure("test_temp1_no_molec")
#     plt.clf()

    # plot_file(True,1)
    # save_figure.save_figure("test_cold_gas1_molec")
    # plt.clf()

plot_test()

