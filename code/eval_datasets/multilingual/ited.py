# -*- coding: utf-8 -*-
# ITED (Indonesian Twitter Emotion Detection)
# - Paper: https://doi.org/10.1109/ICICCS.2019.8822060
# - Data: https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/ITED/Twitter_Emotion_Dataset.csv")

# collect test set here rather than later
_, df = train_test_split(df, test_size=0.2, random_state=42)
df.to_csv("ITED_raw_test_set.csv", index=False)

# construct the the output and instruction for the dataset
def construct_output(row):
    output = row['label']
    return output

def construct_instruction(row):
    instruction = f"""
    This is a data labeling task. The task is emotion classification of a tweet.
    You need to find ONE emotion that best describe the provided tweet among the following 5 categories: anger, happy, sadness, fear, and love.
    Based on the content of the tweet, please choose the most appropriate category as your response.
    Tweet content: {row['tweet']}. Now the tweet ends.
    Please answer: Which one of the five emotion categories best does this tweet: anger, happy, sadness, fear, or love? Answer with one of them.
    """
    return instruction

df['output'] = df.apply(construct_output,axis=1)
df['instruction'] = df.apply(construct_instruction, axis=1)

# a col called language and all values are Indonesian
df['language'] = 'Indonesian'

# a col called subtask and all values are ITED
df['subtask'] = 'ITED'

df_result = df[["instruction", "output", 'language', 'subtask']]
df_result.to_parquet("../../data/ITED/ITED_test.parquet", index=False)