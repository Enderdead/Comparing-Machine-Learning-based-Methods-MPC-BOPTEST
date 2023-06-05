import numpy as np 
import matplotlib.pyplot as plt
import subprocess
import re

def  check_if_image_exists(docker_image):
    cmd_result =  subprocess.run(['docker', 'images'], capture_output=True, text=True).stdout
    return len(re.findall("\\n"+docker_image+" "))>0


def plot_table(title, data):

    title_text = title
    fig_background_color = 'white'
    fig_border = 'steelblue'
    footer_text = ""
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]# Table data needs to be non-numeric text. Format the data
    # while I'm at it.
    cell_text = []
    for row in data:
        cell_text.append([str(x) for x in row])# Get some lists of color specs for row and column headers



    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))# Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
    plt.figure(linewidth=2,
            edgecolor=fig_border,
            facecolor=fig_background_color,
            tight_layout={'pad':1},
            figsize=(11,3)
            )# Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=column_headers,
                        loc='center')# Scaling is the only influence we have over top and bottom cell padding.
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)# Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)# Hide axes border
    plt.box(on=None)# Add title
    plt.suptitle(title_text)# Add footer
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=6, weight='light')# Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plt.draw()
    plt.show()