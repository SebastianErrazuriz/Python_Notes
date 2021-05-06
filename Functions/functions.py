 # -*- coding: utf-8 -*-
"""
Functions to use

@author: Sebastian
"""
import numpy as np
import matplotlib.pyplot as plt



def table_toplot(equipo):
    plt.bar(pos, var_one, color='green', edgecolor='green')
    plt.bar(pos, var_two, color='purple', edgecolor='purple', bottom=var_one)
    plt.bar(pos, var_three, color='yellow', edgecolor='yellow', bottom=np.add(var_one,var_two))

def plot_stacked_bar(axes, ydata, series_labels, xdata=None, 
                     show_values=False, value_format="{}", y_label=None, 
                     colors=None, grid=True, reverse=False, title=None, ymax=None):
    
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    ydata (list)    -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    xdata (list):  -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(ydata[0])
    x = np.arange(len(xdata))

    list_axes = []
    cum_size = np.zeros(ny)

    ydata = np.array(ydata)

    if reverse:
        ydata = np.flip(ydata, axis=1)
        xdata = reversed(xdata)

    for i, row_data in enumerate(ydata):
        color = colors[i] if colors is not None else None
        list_axes.append(axes.bar(x, row_data, bottom=cum_size, 
                                  label=series_labels[i], color=color))
        cum_size += row_data

    if xdata:
        #set xticks and labels
        axes.set_xticks(x) #list of xtick locations.
        axes.set_xticklabels(xdata, rotation = 90) 
    
    if y_label:
        axes.set_ylabel(y_label)
    
    axes.set_yticks(ymax)
        
    if grid:
        axes.grid(True, axis='y')  
        
    if title:
        axes.set_title(title)
        
    if show_values:
        for axis in list_axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                axes.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")
    axes.legend()
    
    
    
def plot_2bar(axes, title, xdata, ydata=[], width=0.40, ylabel=[], color=[], ymax = 0):
    '''Create a barplot for two diferents dataframes columns.
    
    Arguments:
        axes (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        title (string): string of the plot title
        xdata (dataframe col): data frame column, ex: date.
        ydata (list of dataframe col): list of the two dataframes columns that
            we want to compare.
        width (float): width par default 0.40.
        ylabel (list of strings): list of two strings that for the two
            dataframes columns.
        color (list of strings): list of two strings representing the colors.
        ymax (int): in case to specify y max label.
    @author: Sebastian
    '''
    #import numpy and matplotlib
    import numpy as np
    from functions import add_value_labels
    
    # x ticks range
    x = np.arange(len(xdata))
    axes.bar(x + width/2, ydata[0], width, label=ylabel[0], color=color[0])
    axes.bar(x - width/2, ydata[1], width, label=ylabel[1], color=color[1])
    
    axes.set_title(title)
    #st xticks and labels
    axes.set_xticks(x) #list of xtick locations.
    axes.set_xticklabels(xdata, rotation = 90) 
    # set yticks
    axes.set_yticks(ymax)
    axes.legend()
    axes.grid(True, axis='y')
    add_value_labels(axes)     
    
    
def add_value_labels(ax, spacing=5):
    """(Not my code) Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with zero decimal place
        label = "{:.0f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.