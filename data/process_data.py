# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges data.
    
    Loads the individual dataframes, and merges them.
 
    Args:
        messages_filepath (str): Path and name of the 'messages' dataset.
        categories_filepath (str): Path and name of the 'categories' dataset.
    
    Returns:
        df (pd.DataFrame): Dataframe consisting of the two datasets merged into eachother.
    """
    
    # load necessary datasets, merge
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans the raw dataset
    
    Cleaning the dataset by splitting the 'categories' column into 36 individual values, referring to the categories the classification's         output values; then removing duplicates in the dataframe.
 
    Args:
        df (pd.DataFrame): Dataframe consisting of the two datasets merged, unprocessed.
    
    Returns:
        df (pd.DataFrame): The cleaned dataframe.
    """
    # spliting 'categories' column into 36 individual variables
    categories = pd.DataFrame(df.categories.str.split(';'))
    names = []
    row = categories.iloc[0][0]
    for name in row:
        names.append(name[:-2])
    categories[names] = categories.categories.apply(lambda x: pd.Series(str(x).split(",")))
    categories.drop('categories',axis=1,inplace=True)
    
    # obtaining the needed '0' or '1' digit from each string
    for column in categories:
        if column == 'direct_report':
            categories[column] = categories[column].apply(lambda x: str(x)[-3:-2])
        else:
            categories[column] = categories[column].apply(lambda x: str(x)[-2:-1])
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicate entries
    df.drop_duplicates(inplace=True)
    df = df.reset_index().drop('index',axis=1)
    return df


def save_data(df, database_filename):
    """
    Save to df to SQL.
    
    Saves the processed and cleaned database (df) into SQL.
 
    Args:
        df (pd.DataFrame): The dataframe to be saved into SQL.
        database_filename (str): Desired path and name of the SQL database.
    """
    # saving the processed dataframe into a SQL database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()