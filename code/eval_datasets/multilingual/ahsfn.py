# -*- coding: utf-8 -*-
# AHSFN (Arabic COVID-19 Hate Speech & Fake News)
# - Paper: https://doi.org/10.1016/j.procs.2021.05.086
# - Data: https://github.com/MohamedHadjAmeur/AraCOVID19-MFH 
import pandas as pd

data = pd.read_csv('../../data/AHSFN/AraCOVID19-MFH_V1.0.csv', sep=';')
data = data.rename(columns={'Tweet_ID': 'id'})
df = pd.read_csv("ahs_English_raw_test_data.csv")
df = df[['id', 'text']]
data_with_text = pd.merge(data, df, on='id', how='right')

# a. hate speech
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains hate speech. If so, respond "yes". If not, respond "no".
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'yes' or 'no'.
    """
    return insturction

def get_output(row):
    if row['Hate_Speech']=="Hatfull":
          return "yes"
    elif row['Hate_Speech']=="Not Hatfull":
          return "no"

# get a blank dataframe
inst_dataA = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataA = pd.concat([inst_dataA, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-A'], 'language': ['Arabic']})], ignore_index=True)

inst_dataA = inst_dataA.dropna()
# b.vaccine/cure
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains any information or discussion about a cure, a vaccine, or other possible COVID-19 treatments. If so, respond "yes". If not, respond "no".
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'yes' or 'no'.
    """
    return insturction

def get_output(row):
    if row['Talk_About_Cure_or_Vaccine']=="Yes":
          return "yes"
    elif row['Talk_About_Cure_or_Vaccine']=="No":
          return "no"

# get a blank dataframe
inst_dataB = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataB = pd.concat([inst_dataB, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-B'], 'language': ['Arabic']})], ignore_index=True)

inst_dataB = inst_dataB.dropna()

# c.give advice
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below tries to advise people or government institutions. If so, respond "yes". If not, respond "no".
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'yes' or 'no'.
    """
    return insturction

def get_output(row):
    if row['Give_Advice']=="Yes":
          return "yes"
    elif row['Give_Advice']=="No":
          return "no"

# get a blank dataframe
inst_dataC = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataC = pd.concat([inst_dataC, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-C'], 'language': ['Arabic']})], ignore_index=True)

inst_dataC = inst_dataC.dropna()

#d.rise moral
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains encouraging, helpful, and positive speech. If so, respond "yes". If not, respond "no".
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'yes' or 'no'.
    """
    return insturction

def get_output(row):
    if row['Rise_Moral']=="Yes":
          return "yes"
    elif row['Rise_Moral']=="No":
          return "no"

# get a blank dataframe
inst_dataD = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataD = pd.concat([inst_dataD, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-D'], 'language': ['Arabic']})], ignore_index=True)

inst_dataD = inst_dataD.dropna()

# e. news or opinion
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below is news or opinion.
    News: if the tweet report news or a fact.
    Opinion: if the tweet expresses a person's opinion or thoughts.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'News' or 'Opinion'.
    """
    return insturction

def get_output(row):
    if row['News_or_Opinion']=="News":
          return "News"
    elif row['News_or_Opinion']=="Opinion":
          return "Opinion"

# get a blank dataframe
inst_dataE = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataE = pd.concat([inst_dataE, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-E'], 'language': ['Arabic']})], ignore_index=True)

inst_dataE = inst_dataE.dropna()

# f. dialect
def get_instruction(row):
    insturction = f"""Please determine whether the tweet is written in Modern Standard Arabic (MSA), North African dialect, or Middle Eastern dialect.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with MSA, North Africa, or Middle East.
    """
    return insturction

# get a blank dataframe
inst_dataF = pd.DataFrame(columns=['instruction', 'output'])

for i, row in data_with_text.iterrows():
    if row['Dialect'] == """Can't decide""":
        continue
    instruction = get_instruction(row)
    output = row['Dialect']
    inst_dataF = pd.concat([inst_dataF, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-F'], 'language': ['Arabic']})], ignore_index=True)

inst_dataF = inst_dataF.dropna()

# g. blame and negative speech
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains blame, negative, or demoralizing speech. If so, respond "yes". If not, respond "no".
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'yes' or 'no'.
    """
    return insturction

def get_output(row):
    if row['Blame_Negative_Demoralizing_Speech']=="Yes":
          return "yes"
    elif row['Blame_Negative_Demoralizing_Speech']=="No":
          return "no"

# get a blank dataframe
inst_dataG = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataG = pd.concat([inst_dataG, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-G'], 'language': ['Arabic']})], ignore_index=True)

inst_dataG = inst_dataG.dropna()

# h. factual
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains information that can be verified and classified as Fake or Real. Note that this is NOT classifying Fake or Real. It is about determining if the tweet contains information that can be verified.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'Is Not Verifiable' or 'Is Verifiable'.
    """
    return insturction

def get_output(row):
    if row['Factual']=="Is Not Verifiable":
          return "Is Not Verifiable"
    elif row['Factual']=="Is Verifiable":
          return "Is Verifiable"

# get a blank dataframe
inst_dataH = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      instruction = get_instruction(row)
      output = get_output(row)
      inst_dataH = pd.concat([inst_dataH, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-H'], 'language': ['Arabic']})], ignore_index=True)

inst_dataH = inst_dataH.dropna()

# i. worth fact-checking
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains an important claim or dangerous content that maybe be of worth for manual fact-checking.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'Maybe', 'Yes', or 'No'.
    """
    return insturction

# get a blank dataframe
inst_dataI = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      if row['Worth_Fact-checking'] == """Can't decide""":
           continue
      instruction = get_instruction(row)
      output = row['Worth_Fact-checking']
      inst_dataI = pd.concat([inst_dataI, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-I'], 'language': ['Arabic']})], ignore_index=True)

inst_dataI = inst_dataI.dropna()

# j. contain fake info
def get_instruction(row):
    insturction = f"""Please determine if the provided tweet below contains any fake information.
    Tweet: "{row['text']}". Now the tweet ends.
    Please respond with 'Maybe', 'Yes', or 'No'.
    """
    return insturction

# get a blank dataframe
inst_dataJ = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

for i, row in data_with_text.iterrows():
      if row['Contains_Fake_Information'] == """Can't decide""":
           continue
      instruction = get_instruction(row)
      output = row['Contains_Fake_Information']
      inst_dataJ = pd.concat([inst_dataJ, pd.DataFrame({'instruction': [instruction], 'output': [output], 'subtask': ['AHS-J'], 'language': ['Arabic']})], ignore_index=True)

inst_dataJ = inst_dataJ.dropna()

data = pd.concat([inst_dataA, inst_dataB, inst_dataC, inst_dataD,
                  inst_dataE, inst_dataF, inst_dataG, inst_dataH,
                  inst_dataI, inst_dataJ]).reset_index()
data[['instruction', 'output', 'subtask']].to_parquet("../../data/AHSFN/AHS.parquet", index = False)