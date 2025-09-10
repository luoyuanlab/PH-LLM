# MC (COVID-19 Misinformation Communities)
# - Paper: https://doi.org/10.48550/arXiv.2008.00791
# - Data: https://doi.org/10.5281/zenodo.4024154

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
from sklearn.model_selection import train_test_split
from config import client
import math

data = pd.read_csv("../../data/MC/CMU_MisCov19_dataset.csv")
## randomly select 20% as the test set
_, data = train_test_split(data, test_size=0.2, random_state=42)
id_list = data['status_id'].tolist()

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

data = data.rename(columns={"status_id": "id"})
data_with_text = pd.merge(data, df, on='id', how='right')

def get_instruction(text):
    insturction = f"""Please determine if the tweet below meets any of the following conditions. If so, respond with "yes", otherwise respond with "no". The conditions are:
1. The tweet calls out or makes fun of a fake cure, a fake prevention, fake treatment, or a conspiracy theory.
2. The tweet links out to a site that debunks, calls out or makes fun of a fake cure, a fake prevention, fake treatment, or a conspiracy theory.
3. The tweet calls out or make fun of violations of social distancing rules or public health responses.
4. The tweet reports/quotes a (news) story related to consequences of a false fact, fake prevention, fake cure, fake treatment, or conspiracy theory.
5. The tweet reports/quotes a (news) story debunking a false fact, fake prevention, fake cure, fake treatment, or conspiracy theory.
    Please answer with "yes" or "no". You don't need to provide any explanation.
    Tweet: {text}. Now the Tweet ends.
    """
    return insturction

# get a blank dataframe
inst_data1 = pd.DataFrame(columns=['instruction', 'output', 'subtask'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row['text'])
  # output is yes if either annotation1 or annotation2 is "calling out or correction"
  keyword = 'calling out or correction'
  if row['annotation1'] == keyword or row['annotation2'] == keyword:
    output = 'yes'
  else:
    output = 'no'
  inst_data1 = pd.concat([inst_data1, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['MisCOV-A']})], ignore_index=True)

def get_instruction(text):
    insturction = f"""Please determine if the tweet should be classified as conspiracy. If so, respond with "yes", otherwise respond with "no".
A tweet shall be classified as a conspiracy if it endorses a conspiracy story. Some examples of conspiracy themes related to COVID-19 include:
1. It is a bioweapon.
2. Electromagnetic fields and the introduction of 5G wireless technologies led to COVID-19 outbreaks.
3. This was a plan from Gates Foundation to increase the Gates’ wealth.
4. It leaked from the Wuhan Labs or Wuhan Institute of Virology in China.
5. It was predicted by Dean Koontz.
    Tweet: {text}. Now the Tweet ends.
    Please answer with "yes" or "no". You don't need to provide any explanation.
    """
    return insturction

# get a blank dataframe
inst_data2 = pd.DataFrame(columns=['instruction', 'output', 'subtask'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row['text'])
  # output is yes if either annotation1 or annotation2 is "calling out or correction"
  keyword = 'conspiracy'
  if row['annotation1'] == keyword or row['annotation2'] == keyword:
    output = 'yes'
  else:
    output = 'no'
  inst_data2 = pd.concat([inst_data2, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['MisCOV-B']})], ignore_index=True)

def get_instruction(text):
    insturction = f"""Please determine if the tweet should be classified as politics. If so, respond with "yes", otherwise respond with "no".
A tweet shall be classified as politics if the tweet mentions a political individual, institution, or government organization (eg. Congress, Democratic or Republican party), and any of the following conditions are met:
1. The tweet implicitly comments on actions taken by the political actor.
2. The tweet provides commentary on actions taken by the political actor.
    Tweet: {text}. Now the Tweet ends.
    Please answer with "yes" or "no". You don't need to provide any explanation.
    """
    return insturction

# get a blank dataframe
inst_data3 = pd.DataFrame(columns=['instruction', 'output', 'subtask'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row['text'])
  # output is yes if either annotation1 or annotation2 is "calling out or correction"
  keyword = 'politics'
  if row['annotation1'] == keyword or row['annotation2'] == keyword:
    output = 'yes'
  else:
    output = 'no'
  inst_data3 = pd.concat([inst_data3, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['MisCOV-C']})], ignore_index=True)

def get_instruction(text):
    insturction = f"""Please determine if the tweet below meets any of the following conditions. If so, respond with "yes", otherwise respond with "no". The conditions are:
1. The tweet contains clear signs of a satire calling out a fake cure, a fake prevention or a conspiracy.
2. The tweet includes a clear joke about a fake cure, a fake prevention or a conspiracy.
Concretely, this is a tweet where the information in the post is false but is presented using humor, irony, exaggeration, or ridicule to expose and criticize people’s stupidity or vices, particularly in the context of contemporary politics and other topical issues. This kind of post is used to ridicule other false statements or people.
    Tweet: {text}. Now the Tweet ends.
    Please answer with "yes" or "no". You don't need to provide any explanation.
    """
    return insturction

# get a blank dataframe
inst_data4 = pd.DataFrame(columns=['instruction', 'output', 'subtask'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row['text'])
  # output is yes if either annotation1 or annotation2 is "calling out or correction"
  keyword = 'sarcasm or satire'
  if row['annotation1'] == keyword or row['annotation2'] == keyword:
    output = 'yes'
  else:
    output = 'no'
  inst_data4 = pd.concat([inst_data4, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['MisCOV-D']})], ignore_index=True)

def get_instruction(text):
    insturction = f"""Please determine if the tweet below meets any of the following conditions. If so, respond with "yes", otherwise respond with "no". The conditions are:
1. The tweet mention a false fact or prevention against COVID-19 that cannot be verified by the World Health Organization (WHO) or the Centers for Disease Control and Prevention (CDC).
2. The tweet mention a false fact or prevention against COVID-19 that is not supported by a peer-reviewer scientific study, or a preprint from reputable academic sources.
    Tweet: {text}. Now the Tweet ends.
    Please answer with "yes" or "no". You don't need to provide any explanation.
    """
    return insturction

# get a blank dataframe
inst_data5 = pd.DataFrame(columns=['instruction', 'output', 'subtask'])

for i, row in data_with_text.iterrows():
  instruction = get_instruction(row['text'])
  keyword = 'false fact or prevention'
  if row['annotation1'] == keyword or row['annotation2'] == keyword:
    output = 'yes'
  else:
    output = 'no'
  inst_data5 = pd.concat([inst_data5, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['MisCOV-E']})], ignore_index=True)

inst_data = pd.concat([inst_data1, inst_data2, inst_data3, inst_data4, inst_data5], ignore_index=True)
inst_data.to_parquet("../../data/MC/MisCov_test.parquet", index=False)