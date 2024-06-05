import pandas as pd
import numpy as np

# import CPM normalized contig counts
df_cpm = pd.read_csv('Data/CPMs_filtered_integers.txt', sep='\t', index_col=0)
df_cpm = df_cpm.T

# Prepare sheep data for June 13
df13 = pd.read_excel('Data/HiLow sheep CH4 data 13&28 June 2011_for Ioannis_final.xlsx', sheet_name=0)
df13 = df13.ffill()
grouped = df13[df13.columns[[3, 4, 8, 11]]].groupby(by=['Sheep#', 'Metatranscriptome ID']).agg([np.average, 'std'])
grouped.columns = [f'{col[0]}_{col[1]}' for col in grouped.columns]
df13_grouped = grouped.reset_index()
df13_grouped['Date'] = '2011-06-13'

# Prepare sheep data for June 28
df28 = pd.read_excel('Data/Sheep_data.xlsx', sheet_name=1)
df28 = df28.bfill()
grouped = df28[df28.columns[[5, 6, 7, 12]]].groupby(by=['Sheep#', 'Metatranscriptome ID']).agg([np.average, 'std'])
grouped.columns = [f'{col[0]}_{col[1]}' for col in grouped.columns]
df28_grouped = grouped.reset_index()
df28_grouped['Date'] = '2011-06-28'

# combine both dates
sheep_grouped = pd.concat([df13_grouped, df28_grouped], ignore_index=True)

# join metatranscriptome data with label CH4 g/kg DMI_average
#   define features and target
#     features: genes
#     target: CH4 g/kg DMI_average
df_CH4DMI = sheep_grouped[['Metatranscriptome ID', 'CH4 g/kg DMI_average']].set_index('Metatranscriptome ID')
df_gene_CH4DMI = df_cpm.join(df_CH4DMI, how='inner')
data = df_gene_CH4DMI.to_numpy()
X = data[:,:-1]
Y = data[:,-1]

# read in sheep clusters found based on hierarchical clustering of their methane output (Sec. 2.1)
df_clusters = pd.read_csv('Data/clustering_based_clusters.csv').set_index('Metatranscriptome ID')