import fire
import pandas as pd

from sklearn.model_selection import train_test_split


def process(country):
  
  # Load the CSV file
  df = pd.read_csv(f'Ali_Express/data/{country}/train.csv')
  
  # Split the data into 90% training and 10% testing
  train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
  
  # Overwrite the original 'train.csv' with the new train set
  train_df.to_csv(f'Ali_Express/data/{country}/train.csv', index=False)

  # Save the test set
  test_df.to_csv(f'Ali_Express/data/{country}/val.csv', index=False)
  
  # Select clicked data only
  train_clicked_df = train_df[train_df['click']==1]
  
  # Save to df
  train_clicked_df.to_csv(f'Ali_Express/data/{country}/train_clicked.csv', index=False)

  
if __name__ == '__main__':
  fire.Fire(process)