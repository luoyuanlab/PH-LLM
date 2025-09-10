# -*- coding: utf-8 -*-
# WCV (Weibo COVID-19 Vaccine)
# - Paper: https://doi.org/10.2196/27632
# - Data: Not publicly available
import pandas as pd
from sklearn.model_selection import train_test_split

# Weibo-COVID Vaccine (WCV) dataset
df = pd.read_excel("../../data/WCV/新冠疫苗微博训练数据-含2023新增.xlsx")

# collect test set at the beginning, rather than at the end
_, df = train_test_split(df, test_size=0.2, random_state=42)
categories = [
    '17'
]

category_instructions_zh = {
    '17': "是普通用户关于新冠疫苗的发帖。这里指表达了个人经历态度想法等等的帖子，而区别于政府、企业、社区等组织或机器人发送的不包含个人态度帖子。完全无关新冠疫苗的帖子和单纯去转发别人的帖子的也要排除出这一类"
}

def create_instruction_zh(text, category):
    instruction_template = category_instructions_zh[category]
    instruction = f"微博内容：{text}。微博内容结束。\n\n请标记这条微博是否{instruction_template}。Answer yes or no (answer in English)”。"
    return instruction

instructions_df_zh = pd.DataFrame(columns=['instruction', 'output', 'subtask', 'language'])

# Iterate over each row in the original DataFrame
for index, row in df.iterrows():
    text = row['微博内容']
    for category in categories:
        # Create instruction for the category
        if row[category] == 0 and category == '17':
            output = 'yes'
        else:
            output = 'no'
        instruction_zh = create_instruction_zh(text, category)
        instructions_df_zh = pd.concat([instructions_df_zh, pd.DataFrame({'instruction': [instruction_zh], 'output': [output], 'subtask': ['WCV-A'], 'language': ['Chinese']})], ignore_index=True)

# Second level: multiclasses classification
df = df[df['17']==0]

categories = [
    '1.1', '1.3',
    '3.1', '3.2',
    '4.1',
    '8.1',
    '13.1'
]

category_instructions_zh = {
    '1.1': "提到接种新冠疫苗的意愿。这里指的是表达支持、接受或愿意接种新冠疫苗。",
    '1.3': "表达拒绝新冠疫苗的意愿。这类帖子通常会对新冠疫苗接种表达担心，或表达怀疑态度，或表达拒绝接种、反对、不支持新冠疫苗的接种。担心不安全、没效果等也算在这里。",
    '2.1': "提到认为新冠疫苗安全。安全这里指的是认为安全可靠，无不良反应等等",
    '2.2': "提到认为新冠疫苗不安全。不安全这里指的是怀疑新冠疫苗的安全性，认为可能有不良反应，或对健康带来损害等",
    '3.1': "提到认为新冠疫苗有效。有效在这里指的是可产生抗体，或具有预防新冠肺炎的效果等（有效性正面评价）。比如，“可以防感染”、“减轻重症和死亡”等",
    '3.2': "提到认为新冠疫苗无效/效果不好。这类微博会提及怀疑新冠疫苗的有效性，认为新冠疫苗的无效/效果不好，或病毒变异等导致无法产生抗体或预防新冠肺炎等（有效性负面评价）。比如，“预防不了感染”，“打了疫苗还是感染了”，“打了疫苗疾病还是很严重”等",
    '4.1': "提到认为新冠疫苗重要。这里指的是新冠疫苗是重要的、必要的、必须的等等。",
    '8.1': "提到对新冠疫情的高风险感知。这里指的是认为新冠病毒风险高，疫情很严重，对健康危害大。",
    '13.1': "是关于疫苗的负面信息，如疫苗的谣言、反疫苗运动、反智或反科学运动、疫苗负面事件等。",
    '21': "提到用户自己已经接种了新冠疫苗。"
}

# get a dict to map category to subtask, starting from 'B'
subtask_dict = {category: chr(ord('B') + i) for i, category in enumerate(categories)}

# Iterate over each row in the original DataFrame
# assign a subtask to each category, starting from 'B'
for index, row in df.iterrows():
    text = row['微博内容']
    for category in categories:
        if row[category] == 1:
            output = 'yes'
        else:
            output = 'no'
        subtask = subtask_dict[category]
        instruction_zh = create_instruction_zh(text, category)
        instructions_df_zh = pd.concat([instructions_df_zh, pd.DataFrame({'instruction': [instruction_zh], 'output': [output], 'subtask': ['WCV-'+subtask]})], ignore_index=True)
# some issue with llama-factory, so we need to save the data both as the train and as the test, but they are actually both the test set
instructions_df_zh.to_parquet("../../data/WCV/WCV_test.parquet", index=False)