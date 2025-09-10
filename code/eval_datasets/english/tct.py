# TCT (Twitter COVID-19 Test)
# - Paper: https://doi.org/10.2196/26895
# - Data: Not publicly available

import pandas as pd
from sklearn.model_selection import train_test_split 

df = pd.read_excel('../../data/TCT/非伦敦地区抽样结果（核酸检测）计数-原版.xlsx')
df['topic'] = df['topic'].map(lambda x: ' ' + str(x) + ' ')
df['text'] = df['Hit Sentence']
df = df[['text', 'topic', 'attitude']]
_, df = train_test_split(df, test_size=0.2, random_state=42)
df = df.drop_duplicates(subset='text')

# remove the tweets that are not related to personal accounts
categories = ['18']
category_instructions = {'18': 'is either a news report, sent by the government or government officials, sent by companies, advertisement, sent by bot, sent by any other non-personal accounts, retweet (RT/QT) from others without adding personal comments, or not related to coronavirus testing at all.'}

def create_instruction(text, category):
    instruction_template = category_instructions[category]
    instruction = f"""Please read a tweet and follow a data labelling request below.
    Tweet: {text}.
    Data labelling request: Please tell me if this is a tweet that {instruction_template}. 
    If so, answer 'yes'; otherwise, respond with 'no'.
    """
    return instruction

instructions_df = pd.DataFrame(columns=['instruction', 'output', 'category', 'language'])

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    text = str(row['text'])
    for category in categories:
        # Create instruction for the category
        instruction = create_instruction(text, category)
        if row['topic'].find(" "+category+" ")>=0:
            output = 'yes'
        else:
            output = 'no'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output], 'category': ['TCT-A'], 'language': ['English']})], ignore_index=True)

# Now we will create instructions for the categories 5.1 and 16
# remove those with category 18
df = df[~df['topic'].str.contains(' 18 ')]

# lets go
categories = ['5.1', '16']

# get a dictionary to map category to letters, starting from B
category_to_letter = {category: chr(ord('B') + i) for i, category in enumerate(categories)}

# please translate
category_instructions = {
    '5.1': 'expresses understanding, supporting, accepting mass COVID-19 testing',
    '16': 'mentions COVID-19 test for certain subpopulations',
}

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    text = str(row['text'])
    for category in categories:
        # Create instruction for the category
        instruction = create_instruction(text, category)
        if row['topic'].find(" "+category+" ")>=0:
            output = 'yes'
        else:
            output = 'no'
        instructions_df = pd.concat([instructions_df, pd.DataFrame({'instruction': [instruction], 'output': [output], 'category': ['TCT-'+category_to_letter[category]], 'language': ['English']})], ignore_index=True)

instructions_df.to_parquet('../../data/TCT/TCT_test.parquet', index = False)