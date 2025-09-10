#  GHC (Gab Hate Corpus) 
# - Paper: https://doi.org/10.1007/s10579-021-09569-x
# - Data: https://osf.io/edua3

import pandas as pd

df = pd.read_csv('../../data/GHC/ghc_test.tsv', sep='\t')

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

# Assaults on human dignity (HD)
template = """
Please determine if the following Gab post should be classified as assaults on human dignity (HD) or not: {col_value}. Now the post ends. If so, respond with 'yes', otherwise respond with 'no'.
A document should be labeled as assaults on human dignity if it assaults the dignity of group by: asserting or implying the inferiority of a given group by virtue of intelligence, genetics, or other human capacity or quality; degrading a group, by comparison to subhuman entity or the use of hateful slurs in a manner intended to cause harm; the incitement of hatred through the use of a harmful group stereotype, historical or political reference, or by some other contextual means, where the intent of the speaker can be confidently assessed.
In the evaluation of slurs against group identity (race, ethnicity, religion, nationality, ideology, gender, sexual orientation, etc.), we define such instances as hate-based if they are used in a manner intended to wound; this naturally excludes the casual or colloquial use of hate slurs. As an example, the adaptation of the N-slur (replacing the \-er" with \-a") often implies colloquial usage.
Language which dehumanizes targeted persons/groups will also be labeled as HD. In coding dehumanizing rhetoric, we refer coders to Haslam (2006), who developed a model for two forms of dehumanization. In mechanistic forms, humans are denied characteristics that are uniquely human (p. 252). Depriving the other from such traits is considered downward, animalistic comparison. Put another way, the target has been denied the traits that would separate them from animals.
In another form of dehumanization as categorized by Haslam (2006), the target may be denied qualities related to human nature. These characteristics are traits that may not be unique to humans, but define them. These traits will represent the concept's core but may not the same ones that distinguish us from other species" (p. 256). When these traits are denied from the target, this is considered upward, mechanistic dehumanization. The result of denial is often perceiving the target as cold, robotic, and lacking deep-seated core values and characteristics.
Documents which invoke cultural, political, or historical context in order to voice negative sentiment/degradation toward a particular sub-population, empower hateful ideology (hate groups), or reduce the power of marginalized groups, are to be considered HD as well. This would include messages which indicate support for white supremacy (e.g. advocating for segregated societies/apartheid), those which make negative assertions and/or implications about the rights of certain groups (e.g. Immigrants in this country need to go back to their country), and those that reduce the power/agency of particular segments of the population.
Now, please provide a response of either 'yes' or 'no' to the question above. You don't need to provide any explanation.
"""

df2 = apply_template_to_column(df, 'text', template)

# get outputs
df2['output'] = df2['hd'].apply(lambda x: 'yes' if x == 1 else 'no')
df2['subtask'] = 'GHC-A'
df2.output.value_counts()

# Vulgarity/Offensive Language directed at an individual (VO)
template = """
Please determine if the following Gab post should be classified as vulgarity/offensive language directed at an individual (VO) or not: {col_value}. Now the post ends. If so, respond with 'yes', otherwise respond with 'no'. You don't need to provide any explanation.
"""

df3 = apply_template_to_column(df, 'text', template)

# get outputs
df3['output'] = df3['vo'].apply(lambda x: 'yes' if x == 1 else 'no')
df3['subtask'] = 'GHC-B'
df3.output.value_counts()

data = pd.concat([df2[['instruction', 'output', 'subtask']], df3[['instruction', 'output', 'subtask']]], ignore_index=True)
data.to_parquet('../../data/GHC/GHC_test.parquet', index=False)