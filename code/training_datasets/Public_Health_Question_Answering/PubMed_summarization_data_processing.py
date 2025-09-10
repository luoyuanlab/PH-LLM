# PubMed Summarization
# - Paper: https://pubmed.ncbi.nlm.nih.gov/33085945/
# - Data: Not publicly available
import pandas as pd

# PubMedQA
pubmed1 = pd.read_parquet("../../data/Pubmed/1.parquet")
pubmed2 = pd.read_parquet("../../data/Pubmed/2.parquet")
pubmed3 = pd.read_parquet("../../data/Pubmed/3.parquet")
pubmed4 = pd.read_parquet("../../data/Pubmed/4.parquet")
pubmed = pd.concat([pubmed1, pubmed2, pubmed3, pubmed4])

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
def extract_instructions_outputs(conversations):
    instructions = []
    outputs = []
    
    for convo in conversations:
        instruction = next((msg['value'] for msg in convo if msg['from'] == 'human'), None)
        output = next((msg['value'] for msg in convo if msg['from'] == 'agent'), None)
        instructions.append(instruction)
        outputs.append(output)
    
    return instructions, outputs
# Extract instructions and outputs
instructions, outputs = extract_instructions_outputs(pubmed['conversations'])

# Create a new dataframe with the extracted data
pubmed_new = pd.DataFrame({
    'instruction': instructions,
    'output': outputs
})

pubmed_new["instruction"] = pubmed_new["instruction"].map(lambda x: x.replace("[", "").replace("]", "").replace("\', \'", " "))
pubmed_new.to_parquet("../../data/Pubmed/pubmed_merged.parquet")
filtered_pubmed = filter_dataframe(pubmed_new, ["instruction"], keywords)
filtered_pubmed.to_parquet("../../data/Pubmed/Pubmedfiltered_pubmed.parquet")
filtered_pubmed = filtered_pubmed.sample(n=5000, random_state=42)
filtered_pubmed.to_parquet("../../data/Pubmed/Pubmedfiltered_pubmed_sample.parquet")