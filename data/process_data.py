"""
Preprocessing section

"""

#importing relevant libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load two datasets --> messages and categories
    Arguments:
    messages_filepath -> path to csv file that contains
    messages dataset
    categories_filepath -> path to csv file that
    contains categories dataset
    Output:
        df -> combined dataset merged on the id
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    """
    Clean dataset in order to get the right columns

    Arguments:
        Dataframe from the previous step
    Output:
        df -> Combined and cleaned dataset
    """
    #split categories by ';'
    categories = df['categories'].str.split(pat=';',expand=True)
    print(df.head())
    #access the first row of the dataset
    row = categories.iloc[0]
    #define the category name as the entry of the column except the last two characters
    row = [x[:-2] for x in row]
    category_colnames = row
    categories.columns = category_colnames
    print(categories.columns)

    #convert the category columns into int
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    print(categories.head())

    #drop the old category column from the dataset and insert the new colums
    df.drop('categories', axis=1, inplace=True)
    print(df.head())
    pd.concat([df,categories], join='inner', axis=1)
    # drop all duplicates from the dataset
    df.drop_duplicates(inplace=True)
    return df


def save_data(df,database_filename):
    """
    Save the dataframe to database

    Arguments:
        df --> cleaned datasetcategories
        databse_filename --> Path of destination database
    """

    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine,index=False, if_exists="replace")


def main():
    """
    The main function executes the previous steps of the script

    Load function, clean function and save function
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print(df.head())

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