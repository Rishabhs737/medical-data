import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = df['overweight'] = (df['weight'] / (df['height']/100)**2 > 25).astype(int)

# 3
df_gluc_choles = df[['cholesterol', 'gluc']].copy()
df_gluc_choles['cholesterol'] = (df_gluc_choles['cholesterol'] != 1).astype(int)
df_gluc_choles['gluc'] = (df_gluc_choles['gluc'] != 1).astype(int)

df[['cholesterol', 'gluc']] = df_gluc_choles

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars = ['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], var_name='variable',value_name='value')

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    df_cat.rename(columns={'variable': 'Variable', 'value': 'Value', 'count': 'Count'}, inplace=True)

    # 7

    g = sns.catplot(data=df_cat, 
                x='Variable',  
                y='Count',  
                hue='Value',  
                col='cardio',  
                kind='bar', 
               )

   
    g.set_axis_labels('variable', 'total') 

    # 8
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['height'] <= df['height'].quantile(0.975)) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['ap_lo'] <= df['ap_hi'])]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15

    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap = 'gist_heat',vmin = -0.08, vmax=0.24)

    # 16
    fig.savefig('heatmap.png')
    return fig
