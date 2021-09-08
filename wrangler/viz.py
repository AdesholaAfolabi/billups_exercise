import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
#from data_preprocess import remove_excess_space
warnings.filterwarnings('ignore')


def bar_chart(data:pd.DataFrame, plot_column: str, target_column: str, title: str):
    '''
    A bar chart of any column
    Parameters
    -----------
    data: DataFrame or name Series.
        
    plot_column: The name of the column to plot wrt late returns. A string is expected
        
    target_column: The name of the target column in the dataset. A string is expected
            
    title: The title of the bar chart
            
    Returns
    -------
    None
    '''
    df = data.copy()
    occ_df = pd.DataFrame(df[[plot_column, target_column]].loc[df[target_column] == 1].groupby\
                          (plot_column)[target_column].count().reset_index())
    sns.set(rc = {'figure.figsize':(14,8)})
    sns.barplot(x = plot_column, y = target_column, data = occ_df, capsize=.2)
    plt.title(title)
    
def double_plotter(df: pd.DataFrame, column_1: str, column_2: str, plot_kind: str, size: tuple, title: str, rotation: int):
    '''
    A double bar chart of different columns
    Parameters
    -----------
    df: DataFrame or name Series.
        
    column_1: The name of the first column in the dataframe that you want to plot
        
    column_2: The name of the second column in the dataframe that you want to plot
            
    plot_kind: The type of plot e.g bar, barh, etc. Refer to Pandas documentation
    
    size: The size of the chart (2-D)
    
    title: The name of the chart
    
    rotation: How the chart should appear (vertically or horizontally)
            
    Returns
    -------
    The chart
    '''
    return df.groupby(column_1)[column_2].value_counts().unstack().plot(kind = plot_kind,
                                                                        figsize = size, 
                                                                        title = title,
                                                                       rot = rotation)
    
def frequency_plotter(df: pd.DataFrame, column: str, xlabel: str, ylabel: str, size: tuple):
    '''
    A bar chart of any column and a frequency count which denotes activities
    Parameters
    -----------
    df: DataFrame or name Series.
        
    xlabel: The name of the X-axis label. A string is expected
        
    ylabel: The name of the Y-axis label. A string is expected
            
    size: The size of the chart
            
    Returns
    -------
    None
    '''
    data = df.copy()
    data = data[column].value_counts()
    x = data.values
    y = data.index
    plt.figure(figsize = (size))
    splot = sns.barplot(x, y, orient='h')
    for p in splot.patches:
        width = p.get_width()
        plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),
             '{:1.2f}'.format(width),
             ha='center', va='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_pie(data: pd.DataFrame, target_value: int, exp: tuple, colors: tuple, title: str, size: tuple):
    '''
    A pie chart of any dataframe column distribution.
    Parameters
    -----------
    data: DataFrame or name Series.
        
    target_value: The name of the target column in the dataset. An int is expected
        
    exp: PyChart explode size
        
    color: The expected color of the pie chart 
    
    title: The title of the pie chart
    
    size: The expected size of each slice of the pie
        
    Returns
    -------
    None
    '''
    
    df = data.copy()
    x = df['checkout_date_returned_day'].loc[df['target'] == target_value].unique()
    y = df['checkout_date_returned_day'].loc[df['target'] == target_value].value_counts()
    wp = { 'linewidth' : 2, 'edgecolor' : "black" }
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%".format(pct, absolute)
    fig, ax = plt.subplots(figsize = size)
    wedges, texts, autotexts = ax.pie(y, 
                                  autopct = lambda pct: func(pct, y),
                                  explode = exp, 
                                  labels = x,
                                  shadow = True,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="black"))
    plt.title(title)
    
def word_cloud(words: list):
    
    '''
    Word Count visualization for text features
    
    Parameters
    -----------
    words: The words to be plotted
        
    Returns
    -------
    None
    '''
    
    all_text = " ".join([desc for desc in words])
    # Initialize wordcloud object
    wc = WordCloud(background_color='white', max_words=50)

    # Generate and plot wordcloud
    plt.imshow(wc.generate(all_text))
    plt.axis('off')
    plt.show()