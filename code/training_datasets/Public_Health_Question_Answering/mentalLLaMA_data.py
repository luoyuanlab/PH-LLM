# MentalLLaMA QA
# - Paper: https://arxiv.org/abs/2309.13567
# - Data: https://github.com/SteveKGYang/MentalLLaMA/tree/main/train_data
import pandas as pd

dr = pd.read_csv('../../data/MentalLLaMA/instruction_data/DR/train.csv')
dreaddit = pd.read_csv('../../data/MentalLLaMA/instruction_data/dreaddit/train.csv')
Irf = pd.read_csv('../../data/MentalLLaMA/instruction_data/Irf/train.csv')
MultiWD = pd.read_csv('../../data/MentalLLaMA/instruction_data/MultiWD/train.csv')
SADdata = pd.read_csv('../../data/MentalLLaMA/instruction_data/SAD/train.csv')

data = pd.concat([dr.sample(n=1000, random_state=42), 
                  dreaddit.sample(n=2000, random_state=42),
                  Irf.sample(n=3000, random_state=42),
                  MultiWD.sample(n=3000, random_state=42),
                  SADdata.sample(n=1000, random_state=42)])

data = pd.DataFrame({"instruction":data['query']+ " Apart from the answer itself, you should also mention your reasoning.", "output":data["gpt-3.5-turbo"]})
data.to_parquet("../../data/MentalLLaMA/mental_llama_sample.parquet", index = False)