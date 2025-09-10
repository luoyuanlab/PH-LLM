# CAVES (A Dataset to facilitate Explainable Classification and Summarization of Concerns towards COVID Vaccines) 
# - Paper: https://arxiv.org/pdf/2204.13746 
# - Data: https://github.com/sohampoddar26/caves-data/blob/main/labelled_tweets/csv_labels/test.csv

import pandas as pd

df = pd.read_csv('../../data/CAVES/test.csv')

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
    data = df.copy()
    data['instruction'] = data[col_name].apply(lambda x: template.format(col_value=x))
    return data

def check_keyword_in_column(df, col_name, keyword):
    """
    This function checks if a specific keyword exists in each row of the specified column.
    It creates a new column 'contains_keyword' with 'yes' if the keyword is found, and 'no' otherwise.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col_name (str): The column name to check for the keyword.
    keyword (str): The keyword to search for in the column.

    Returns:
    pd.DataFrame: The DataFrame with a new column 'contains_keyword' indicating 'yes' or 'no' for each row.
    """
    
    # Check if the keyword is in each value of the column
    data = df.copy()
    data['output'] = data[col_name].apply(lambda x: 'yes' if keyword in str(x) else 'no')
    return data

# Define the template
template = """Please determine if the tweet below indicates COVID is not dangerous, vaccines are unnecessary, or that alternate cures (such as hydroxychloroquine) are better. 
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df1 = apply_template_to_column(df, 'text', template)
df1 = check_keyword_in_column(df1, 'labels', 'unnecessary')
df1['subtask'] = 'CAVES-A'

# Define the template
template = """Please determine if the tweet below is against mandatory vaccination and talks about their freedom
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df2 = apply_template_to_column(df, 'text', template)
df2 = check_keyword_in_column(df2, 'labels', 'mandatory')
df2['subtask'] = 'CAVES-B'

# Define the template
template = """Please determine if the tweet below indicates that the Big Pharmaceutical companies are just trying to earn money, or is against such companies in general because of their history
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df3 = apply_template_to_column(df, 'text', template)
df3 = check_keyword_in_column(df3, 'labels', 'pharma')
df3['subtask'] = 'CAVES-C'

# NOTE: remove Conspiracy because there are too many of them
# Define the template
template = """Please determine if the tweet below expresses concerns that the governments / politicians are pushing their own agenda though the vaccines
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df4 = apply_template_to_column(df, 'text', template)
df4 = check_keyword_in_column(df4, 'labels', 'political')
df4['subtask'] = 'CAVES-D'

# NOTE: Country not included because of low prevalence
# Define the template
template = """Please determine if the tweet below expresses concerns that the vaccines have not been tested properly, have been rushed or that the published data is not accurate
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df5 = apply_template_to_column(df, 'text', template)
df5 = check_keyword_in_column(df5, 'labels', 'rushed')
df5['subtask'] = 'CAVES-E'

# Define the template
template = """Please determine if the tweet below expresses concerns about the side effects of the vaccines, including deaths caused.
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df7 = apply_template_to_column(df, 'text', template)
df7 = check_keyword_in_column(df7, 'labels', 'side-effect')
df7['subtask'] = 'CAVES-F'

# Define the template
template = """Please determine if the tweet below expresses concerns that the vaccines are ineffective, not effective enough, or are useless.
    If so, respond with 'yes'. Otherwise, respond with 'no'.
    Do not explain your rationale. Please directly respond with 'yes' or 'no'.
    Tweet: {col_value}
    Please respond with 'yes' or 'no'.
    """

df8 = apply_template_to_column(df, 'text', template)
df8 = check_keyword_in_column(df8, 'labels', 'ineffective')
df8['subtask'] = 'CAVES-G'

# Religious not included because of low prevalence in the sample
data = pd.concat([df1, df2, df3, df4, df5, df7, df8], axis=0)
data = data[['instruction', 'output', 'subtask']]
data.to_parquet('../../data/CAVES/CAVES_test.parquet', index = False)