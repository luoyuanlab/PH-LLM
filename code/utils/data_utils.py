import pandas as pd
from sklearn.utils import resample
def balance_classes(data, target_size, column_name):
   """
   Balances the dataset by downsampling the majority classes and upsampling the minority classes.


   Parameters:
   - data (pd.DataFrame): The input DataFrame.
   - target_size (int): The number of samples desired for each class.
   - column_name (str): The column name that contains class labels.


   Returns:
   - pd.DataFrame: A new DataFrame with balanced classes.
   """
   # Group the data by the class column
   grouped = data.groupby(column_name)


   dfs = []
   for class_label, group in grouped:
       if len(group) > target_size:
           # Downsample the majority class
           group = resample(group, replace=False, n_samples=target_size, random_state=42)
       else:
           # Upsample the minority class
           group = resample(group, replace=True, n_samples=target_size, random_state=42)
       dfs.append(group)


   # Concatenate all DataFrames to get the balanced dataset
   balanced_data = pd.concat(dfs)
   return balanced_data

def downsample_to_minority_class(df, target_column):
    # Get the size of the minority class
    class_counts = df[target_column].value_counts()
    min_class_size = class_counts.min()

    # Downsample each class to the size of the minority class
    downsampled_dfs = []
    for class_value in class_counts.index:
        class_df = df[df[target_column] == class_value]
        downsampled_df = resample(class_df,
                                  replace=False,  # sample without replacement
                                  n_samples=min_class_size,  # to match minority class size
                                  random_state=123)  # reproducible results
        downsampled_dfs.append(downsampled_df)

    # Combine all the downsampled dataframes
    df_downsampled = pd.concat(downsampled_dfs)

    return df_downsampled