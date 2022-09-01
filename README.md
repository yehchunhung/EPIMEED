# EPIMEED

In recent years, more and more people choose to use text-based platforms for their mental health support. Thus, building empathetic chatbots has been a popular research topic aiming at generating syntactically and emotionally appropriate responses. In this work, we develop multi-turn empathetic dialog models which can not only recognize human emotions but also rely on the well defined cognitive framework to assess expressed empathy in texts.

The summary of this work can be seen here [Slide version](https://docs.google.com/presentation/d/1WjodJYjNGu9j28TGGo3upg2G1z2S11Cbwy_hh1cNWU4/edit?usp=sharing) and [Poster version](https://drive.google.com/file/d/15awdf-yebXaNvjqd0vzsgG9Y1lE6cUye/view?usp=sharing).

## Table of Contents

- [Files Description](#files-description)
- [Dataset](#dataset)
- [Steps to Reproduce Result](#steps-to-reproduce-result)
- [References](#references)

## Files Description

There are many files used in this work. Here we point out the mainly used ones. 

- `ed`, `osed`, `red`: These 3 folders contain the Python codes for building models with regard to the corresponding datasets `ED`, `OSED`, and `RED` described in this work.
- `utils` and `dataset`: The folders contain the utility functions for modeling and summarizing text dialogs.
- `python-scripts-red-chat-mturk`: This folder is mainly used for the human evaluation, as we'll let human raters to judge the quality of the generated responses. 

## Dataset

The dataset for this project is referred to our prior work, [RED](https://github.com/yehchunhung/RED). To fit our computing resources, we further summarized the text dialogs and filtered to accomodate only the dialogs within 100 tokens. The final version of our used dataset can be seen as followed. Note that each dataset folder may contain 5 files, where `uttrs.txt` stores the selected dialogs while the rest stores the encoded information about text dialogs, emotions, communication levels, and the associated mask tokens.

- Training: [Google Drive](https://drive.google.com/drive/folders/1Mappsxy3kvAprqi2c9TvSbP5o8fp6XEF?usp=sharing)
- Validation: [Google Drive](https://drive.google.com/drive/folders/1ax-JoKRbZuiXf1eP-bRc231E_Bn7VKgA?usp=sharing)
- Testing: [Google Drive](https://drive.google.com/drive/folders/1E_A9KcxH3vK3358lUPeW4i48h54ljhUD?usp=sharing)

## Steps to Reproduce Result

TODO

## References

1. Yubo Xie, Pearl Pu. “Empathetic Dialog Generation with Fine-Grained Intents”. In: Proceedings of the 25th Conference on Computational Natural Language Learning. Online: Association for Computational Linguistics, Nov. 2021.

2. Ashish Sharma, Adam S. Miner, David C. Atkins, and Tim Althoff. A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support. 2020.
