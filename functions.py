# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:26:02 2017

@author: auracoll
"""
import fancyimpute 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# format and section off value_counts for each variable
for val in df.columns: print('# ', val, '\n', 'df', '["', val, '"]',
                             '.value_counts(dropna=False)', '\n',
                             sep='')

# better version of previous function. must be run from cmd.
def EDA(df, name):

    '''
    name == string version of df's name
    proviudes count, unique
        to use:
        1. run cmd as administrator
        2. python scriptpath.py > Output/dfnameEDA.txt
        3. Enjoy the fruits!!!!!
    '''

    df.name = name
    print('#{}\n'.format(df.name))
    for col in df.columns:
        print('#{}\n'.format(col))
        print(df[col].describe())
        print('\n')
        print(df[col].value_counts(dropna=False))
        print('\n')

def impute(data, **kwargs):
    ### Impute missing values | kwargs from MICE args
    
    # can add impute_method=random (or other) to MICE
    impute_missing = data
    impute_missing_cols = list(impute_missing)
    filled_soft = fancyimpute.MICE(**kwargs).complete(np.array(impute_missing))
    results = pd.DataFrame(filled_soft, columns = impute_missing_cols)
    assert results.isnull().sum().sum() == 0, 'Not all NAs removed'
    return results

def nullval(df) :
    ### compare null with observed values
    
    A = df.isnull().sum().sum()
    B = df.notnull().sum().sum()
    print('total NA vals:       ' + str(A))
    print('total observed vals: ' + str(B))
    print('total cells in set:  ' + str(A + B))

def percent_categorical(item, df, grouper='Active Status') :
    # plot categorical responses to an item ('column name')
    # by percent by group ('diff column name w categorical data')
    # select a data frame (default is IA)
    # active vs term is default grouper
    
    # create df of item grouped by status
    grouped = df.groupby(grouper)[item]
        # count frequencies by group
    grpcount = grouped.count()

    # convert to percentage by group rather than total count
    PercGroup = (grouped.value_counts(normalize=True)
                # rename column 
                .rename('percentage')
                # multiple by 100 for easier interpretation
                .mul(100)
                # change order from value to name
                .reset_index()
            .sort_values(item))
    
    # create plot
    fig, ax = plt.subplots()
    PercPlot = sns.barplot(x=item,
                         y='percentage',
                         hue=grouper,
                         data=PercGroup,
                         palette='RdBu',
                         ax=ax).set_xticklabels(
                                 labels = PercGroup[item
                                      ].value_counts().index.tolist(), rotation=90)
    
     # get the handles and the labels of the legend
    # these are the bars and the corresponding text in the legend
    thehandles, thelabels = ax.get_legend_handles_labels()
    # for each label, add the total number of occurences
    # you can get this from groupcount as the labels in the figure have
    # the same name as in the values in column of your df
    for counter, label in enumerate(thelabels):
        # the new label looks like this (dummy name and value)
        # 'XYZ (42)'
        thelabels[counter] = label + ' ({})'.format(grpcount[label])
    # add the new legend to the figure
    ax.legend(thehandles, thelabels)
    #show plot
    return fig, ax, PercPlot

def groupdifs(data, groups='Active Status'):
    # create group comparison of Mean, SD, and N on variables
    # data = df, groups = column name (maybe)
    # create list of all columns, then list of all columns less group col
    AllCols = list(data.columns.values)
    NumbCols = list(AllCols)
    NumbCols.remove(groups)
    # values from groups
    grpnames = list(data[groups].unique())
    # compare active versus terms on the Voice data
    # not sure why, but need to transpose
    Summary_Stats = data.groupby(groups)[NumbCols].agg(
            ['count','mean','std']).transpose()
     # unstack the last operation to create stats columns for each numerical variable
    newstats = Summary_Stats.unstack()
    ### create d-values
    # create pooled sd
    sd1 = newstats.loc[:, (grpnames[0], 'std')]
    sd2 = newstats.loc[:, (grpnames[1], 'std')]
    newstats['pooled_sd'] = np.sqrt((sd1**2 + sd2**2) /2)
    # add new column with D calculation
    newstats['d_value'] = (newstats.loc[:, (grpnames[1], 'mean') # M1
    ] - newstats.loc[:, (grpnames[0], 'mean')]                   # -M@
    ) / newstats.pooled_sd                                        # /SDpooled
    ### sort by the absolute value of cohen's d
    # create new column of absolute d values
    newstats['abs_d'] = abs(newstats.iloc[:,-1])
    # sort by that column
    newstats.sort_values('abs_d', ascending= False, inplace=True)
    del(newstats['abs_d'])
    return newstats
