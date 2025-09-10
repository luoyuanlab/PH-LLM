# OpenOrca
# - Paper: https://doi.org/10.48550/arXiv.2306.02707
# - Data: https://huggingface.co/datasets/Open-Orca/OpenOrca
# Here we only presented the code for collecting public health-related subset from Orca. 
# We also randomly sampled a subset of GPT-4 Orca as our training set but unfortunately we cannot find the code for the subsampling process.

import pandas as pd

orca = pd.read_parquet("../../data/OpenOrca/1M-GPT4-Augmented.parquet")
orca_gpt3 = pd.read_parquet("../../data/OpenOrca/3_5M-GPT3_5-Augmented.parquet")

def filter_dataframe(df, columns, keywords):
    pattern = '|'.join(keywords)
    mask = df[columns].apply(lambda x: x.str.contains(pattern, case=False, na=False, regex=True)).any(axis=1)
    filtered_df = df[mask]
    return filtered_df

# public health keywords
keywords = ["public health", "global health", "vaccine hesitancy", "vaccine confidence", 
            "vaccine uptake", "vaccine acceptance", "immunization rate", "vaccine coverage",
            "vaccination acceptance", "vaccination intent", "depression", "anxiety", 
            "mental health", "pandemic", "epidemic", "outbreak", "misinformation", 
            "rumor", "rumour", "infodemic", "health equity", "social determinants of health", 
            "health promotion", "chronic disease management", "health communication", 
            "community health", "health disparity", "health policy", "environmental health", 
            "occupational health", "health education", "health services research", 
            "health system", "substance abuse", "tobacco control", "health economics", 
            "quality-adjusted life year", "disability-adjusted life year", 
            "cost-effectiveness analysis", "socioeconomic status", "relative risk", 
            "odds ratio", "incidence rate ratio", "herd immunity", "health belief model", 
            "years lived with disability", "cross-sectional", "cohort study", 
            "ecological study", "effective reproductive number", 
            "effective reproduction number", "basic reproduction number", 
            "basic reproductive number", "confounding variable", "healthy worker effect", 
            "health technology assessment", "ecologic fallacy", "morbidity", 
            "global burden of disease", "one health", "telehealth", "health inequality", 
            "social medicine", "infant mortality", "digital health", "digital medicine", 
            "mhealth", "e-health", "m-health", "immunization program", "pharmacovigilance", 
            "universal health coverage", "health disparities", "health disparity", "climate change", "endemic",
            "quality-of-life", "risk factor", "cdc", "community-based participatory research",
            "community health assessment", "environmental hazard", "infectious disease", 
            "population health", "quality of life", "quarantine", "underserved population",
            "lmic", "low- and middle-income country", "low- and middle-income countries", "years of potential life lost", 
            "protective factor", "sporadic disease", "world health oganization"]

filtered_orca = filter_dataframe(orca, ["question", "response"], keywords)
filtered_orca.to_parquet("../../data/OpenOrca/filtered_orca_gpt4.parquet") 
filtered_orca.to_parquet("../../data/OpenOrca/filtered_orca_gpt4_sample.parquet")
filtered_orca_gpt3 = filter_dataframe(orca_gpt3, ["question", "response"], keywords)
filtered_orca_gpt3.to_parquet("../../data/OpenOrca/filtered_orca_gpt3.parquet")
filtered_orca_gpt3.to_parquet("../../data/OpenOrca/filtered_orca_gpt3_sample.parquet")
filtered_orca_gpt3 = pd.read_parquet("../../data/OpenOrca/filtered_orca_gpt3_sample.parquet")
filtered_orca = pd.DataFrame({'system_prompt':filtered_orca['system_prompt'], 'instruction':filtered_orca['question'], 'output': filtered_orca['response']})
filtered_orca_gpt3 = pd.DataFrame({'system_prompt':filtered_orca_gpt3['system_prompt'], 'instruction':filtered_orca_gpt3['question'], 'output': filtered_orca_gpt3['response']})
merged = pd.concat([filtered_orca, filtered_orca_gpt3])
merged = merged.sample(frac=1, random_state=42)
merged.to_parquet("../../data/OpenOrca/public_health_openorca.parquet", index=False)
