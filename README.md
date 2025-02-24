# Understanding Rumen Methanogen Interactions in Sheep

Methane emissions from livestock pose a significant environmental challenge, particularly in Aotearoa New Zealand (NZ). Chemical inhibitors such as feed additives or vaccines could help lower methane emissions. However, their successful development has been hindered by a limited understanding of the complex interactions among the microorganisms in the rumen (forestomach).

This study serves as a proof-of-concept to explore the potential of using metatranscriptome data to understand the genetic basis of microbial interactions in the rumen and identify potential inhibitor targets. 

We analyzed a small dataset of 10 sheep emitting different levels of methane. We employed various statistical and machine learning techniques to uncover promising contigs (continuous sequences of DNA) linked to methane output. Despite the limited sample size, our findings revealed new insights into microbial mechanisms, validated by domain experts. These preliminary results suggest that expanding the dataset and integrating advanced techniques could enhance our understanding of the complex microbial interactions in the rumen, ultimately contributing to the development of effective strategies to mitigate methane emissions in livestock.


# Where to find what

We have organized this repository similar to the paper with individual Jupyter notebooks for each subsection in our Analysis section:
- [Data Preprocessing](data_preprocessing.py)
- [Hypothesis and Data Validaton](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/1_Hypothesis%20and%20Data%20Validaton.ipynb)
- [Narrowing Down the Search](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/2_Narrowing%20Down%20the%20Search.ipynb)
- [Understanding Patterns](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/3_Understanding%20Patterns.ipynb)
- [Identifying Causal Relationships in the Rumen](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/4_Identifying%20Causal%20Relationships%20in%20the%20Rumen.ipynb)
- [Dataset: Contig Counts](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/Data/CPMs_filtered_integers.txt)
- [Dataset: Methane Output](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/Data/Sheep_data.xlsx)

# Funding

This research has been funded by the Ministry of Business, Innovation \& Employment, New Zealand (MBIE number C10X2201) awarded to XXX. XXX and XXX have received research funding from AgResearch New Zealand. 
