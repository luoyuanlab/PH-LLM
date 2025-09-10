# Ethos (Online Hate Speech Detection)
# - Paper: https://doi.org/10.1007/s40747-021-00608-2
# - Data: https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../../data/Ethos/Ethos_Dataset_Binary.csv', sep=';', encoding='utf-8')
_, df = train_test_split(df, test_size=0.2, random_state=42)

def apply_template_to_column(df, col_name, template):
    """
    This function applies a template to a specific column in the DataFrame and generates a new column called 'instruction'.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col_name (str): The column name to which the template should be applied.
    template (str): A template string that will be applied to each value in the column.
                    Use curly braces {col_value} as a placeholder for the column values.

    Returns:
    pd.DataFrame: The DataFrame with a new column 'instruction' containing the result of applying the template.
    """
    
    # Apply the template to each value in the column and store it in a new column called 'instruction'
    df['instruction'] = df[col_name].apply(lambda x: template.format(col_value=x))
    
    return df

df = apply_template_to_column(df, 'comment', 'Please classify if the following social media post contain hate speech: {col_value}. Now the post ends. Hate speech is a form of insulting public speech directed at specific individuals or groups of people on the basis of characteristics, such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity. Please response with "Yes" if the post contains hate speech, and "No" if it does not.')

# if isHate > 0.5, then it is hate speech; else it is not hate speech
df['output'] = df['isHate'].apply(lambda x: "Yes" if x >= 0.5 else "No")
df['subtask'] = 'Ethos'

df[['instruction', 'output', 'subtask']].to_parquet('../../data/Ethos/Ethos_test.parquet', index=False)  # Save to parquet file