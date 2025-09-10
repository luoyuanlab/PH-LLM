# -*- coding: utf-8 -*-
# MAT (Misinformation on Arabic Twitter) 
# - Paper: https://doi.org/10.48550/arXiv.2101.05626
# - Data: https://github.com/SarahAlqurashi/COVID-19-Arabic-Tweets-Dataset
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
from config import client

data = pd.read_csv('../../data/MAT/COVID19-Misinformation-Dataset.txt')
## randomly select 20% as the test set
_, data = train_test_split(data, test_size=0.2, random_state=42)
id_list = data['id'].tolist()

# Get Tweets
def get_tweets_in_batches(client, id_list, batch_size=99):
    all_tweets = []
    
    #calculate the batches needed
    num_batches = math.ceil(len(id_list) / batch_size)
    
    for i in range(num_batches):
        time.sleep(60)
        start = i * batch_size
        end = min((i + 1) * batch_size, len(id_list))
        batch_ids = id_list[start:end]
        
        response = client.get_tweets(batch_ids, tweet_fields=["text"])
        
        if response.data:
            for tweet in response.data:
                all_tweets.append({"id": tweet.id, "text": tweet.text})
    
    # get a blank dataframe
    df = pd.DataFrame(all_tweets)
    return df

df = get_tweets_in_batches(client, id_list)
data_with_text = pd.merge(data, df, on='id', how='right')

#categorise a given tweet text into either being a personal narrative (33.3%) or news (66.7%)
# construct instruction
def get_instruction(row):
    insturction = f"""Please respond 'yes' if the provided tweet below contains misinformation. Otherwise, respond 'no'.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'yes' or 'no'.
    """
    return insturction
  
def get_output(row):
      if row['label']==0:
            return 'no'
      return 'yes'

# get a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row)
  output = get_output(row)
  inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['MAT'], 'language': ['Arabic']})], ignore_index=True)
inst_data.to_parquet("../../data/MAT/MAT_test.parquet", index=False)