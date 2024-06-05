# Understanding Rumen Methanogen Interactions in Sheep

Methane emissions from livestock pose a significant environmental challenge, particularly in Aotearoa New Zealand (NZ). Chemical inhibitors such as feed additives or vaccines could help lower methane emissions. However, their successful development has been hindered by a limited understanding of the involved mechanisms.

This study serves as a proof-of-concept to explore the potential of using metatranscriptome data to understand microbial interactions in the rumen and identify molecular drivers of methane production. 
We analyzed a small dataset from 10 sheep, employing various statistical and machine learning techniques to uncover promising contigs (continuous sequences of DNA) linked to methane output. 

Despite the limited sample size, our findings revealed new insights into microbial mechanisms, validated by domain experts. These preliminary results suggest that expanding the dataset and integrating advanced techniques could enhance our understanding, ultimately contributing to the development of effective strategies to mitigate methane emissions in livestock.

# Citation

If you want to use this implementation or cite any of our findings in your publication, please cite the following ICDM paper (currently under review):
```
Katharina Dost, Steffen Albrecht, Paul Maclean, Sandeep Gupta, and Jörg Wicker.
"Understanding Rumen Methanogen Interactions in Sheep."
In: Forthcoming, 2024.
```

Bibtex:
```
@INPROCEEDINGS {dost2024understanding,
author = {Katharina Dost and Steffen Albrecht and Paul Maclean and Sandeep Gupta and J\"org Wicker},
title = {Understanding Rumen Methanogen Interactions in Sheep},
year = {2024},
booktitle = {Forthcoming}
}
```

# Where to find what

We have organized this repository similar to the paper with individual Jupyter notebooks for each subsection in our Analysis section:
- [Data Preprocessing](data_preprocessing.py)
- [Hypothesis and Data Validaton](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/1_Hypothesis%20and%20Data%20Validaton.ipynb)
- [Narrowing Down the Search](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/2_Narrowing%20Down%20the%20Search.ipynb)
- [Understanding Patterns](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/3_Understanding%20Patterns.ipynb)
- [Identifying Candidates for Mitigating Measures](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/4_Identifying%20Candidates%20for%20Mitigating%20Measures.ipynb)
- [Dataset: Contig Counts](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/Data/CPMs_filtered_integers.txt)
- [Dataset: Methane Output](https://github.com/KatDost/Sheep_Methane_Paper/blob/main/Data/Sheep_data.xlsx)
