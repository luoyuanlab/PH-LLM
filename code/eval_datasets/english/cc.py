# CC (COVID category – personal vs news) 
# - Paper: https://doi.org/10.3389/frai.2023.1023281
# - Data: https://github.com/digitalepidemiologylab/covid-twitter-bert/tree/master/datasets/covid_category
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from config import client

data = pd.read_csv("../../data/CC/covid_category.csv")
## randomly select 20% as the test set
_, data = train_test_split(data, test_size=0.2, random_state=42)
id_list = data['id'].tolist()
# Get Tweets
def get_tweets_in_batches(client, id_list, batch_size=99):
    all_tweets = []
    
    #calculate the batches needed
    num_batches = math.ceil(len(id_list) / batch_size)
    
    for i in range(num_batches):
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

def get_instruction(row):
    insturction = f"""Please categorize a given tweet text into either being a personal narrative or news.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with "news" or "personal".
    """
    return insturction


# get a blank dataframe
inst_data = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row)
  output = row['label'].replace("category_", "")
  inst_data = pd.concat([inst_data, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['CC'], 'language': ['English']})], ignore_index=True)

inst_data.to_parquet("../../data/CC/cc_test.parquet", index=False)
