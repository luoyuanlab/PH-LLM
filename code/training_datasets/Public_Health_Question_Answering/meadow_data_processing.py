# Meadow (Medical Flashcards)
# - Paper: https://doi.org/10.48550/arXiv.2304.08247
# - Data: https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards
import pandas as pd


df = pd.read_json("hf://datasets/medalpaca/medical_meadow_medical_flashcards/medical_meadow_wikidoc_medical_flashcards.json")
df = df[['input', 'output']]
df = df.applymap(lambda x: None if x == '' else x)
df = df.dropna()

def filter_dataframe(df, columns, keywords):
    pattern = '|'.join(keywords)
    mask = df[columns].apply(lambda x: x.str.contains(pattern, case=False, na=False, regex=True)).any(axis=1)
    filtered_df = df[mask]
    return filtered_df


# Public Health Keywords for Instruction Tuning Datasets

public_health_keywords = ["public health", "global health", "vaccine hesitancy", "vaccine confidence", 
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

instruction_data = filter_dataframe(df, ["input", "output"], public_health_keywords)
instruction_data = pd.DataFrame({"instruction": instruction_data['input'], "output": instruction_data['output']})
instruction_data = instruction_data.sample(n=1400, random_state=42)
instruction_data.to_parquet('../../data/Meadow/PublicHealthMeadow.parquet', index = False)
