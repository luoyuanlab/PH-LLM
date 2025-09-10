# MedMCQA (Medical MCQA)
# - Paper: https://doi.org/10.48550/arXiv.2203.14371
# - Data: https://huggingface.co/datasets/openlifescienceai/medmcqa
import pandas as pd

MedMCQA = pd.read_parquet('../../data/MedMCQA/train-00000-of-00001.parquet')

spmdata = MedMCQA[MedMCQA['subject_name']=="Social & Preventive Medicine"]
psychiatry = MedMCQA[MedMCQA['subject_name']=="Psychiatry"]
MedMCQASample = pd.concat([spmdata, psychiatry])
MedMCQASample = MedMCQASample.sample(n=15000, random_state=42)
def merge_into_instruction(row):
    return f"""Please answer the following multiple-choice question relevant to clinical or public health: {row['question']}
Choose one of the following options: A: {row['opa']}, B: {row['opb']}, C: {row['opc']}, D: {row['opd']}.
Answer with A, B, C, or D. Your answer is: """

MedMCQASample['instruction'] = MedMCQASample.apply(merge_into_instruction, axis=1)


cop_to_output = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
MedMCQASample['output'] = MedMCQASample['cop'].map(cop_to_output)

Instruction_data = MedMCQASample[['instruction', 'output']]
Instruction_data.to_parquet("../../data/MedMCQA/PublicHealthMCQA.parquet", index=False)