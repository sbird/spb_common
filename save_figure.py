"""Module with one function which saves a figure"""
import matplotlib.pyplot as plt
import matplotlib.backends

def save_figure(path):
    """Saves the figure, automatically determining file extension"""
    bk=matplotlib.backends.backend
    if path == "":
        return
    elif bk == 'TkAgg' or bk == 'Agg' or bk == 'GTKAgg' or bk == 'Qt4Agg':
        path = path+".png"
    elif bk == 'PDF' or bk == 'pdf':
        path = path+".pdf"
    elif bk == 'PS' or bk == 'ps':
        path = path+".ps"
    return plt.savefig(path,bbox_inches='tight')

