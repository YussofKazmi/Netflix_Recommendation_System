import pandas as pd
import os

def load_and_view_df(url):
    df = pd.read_csv(url)
    #print(df.head())
    
    print("Dataframe columns: ", df.columns)    
    print("Dataframe Size: ", len(df))
    print("Number of unique titles: ", len(df['title'].unique()) )
    return df


def clean_movies_df(df):
    # drop budget, id, production_countries, crew
    columns_to_drop = ['budget', 'id', 'production_countries', 'crew']
    
    df = df.drop(columns_to_drop, axis = 1)
    
    print(df.head())
    print("Dataframe columns: ", df.columns)    
    print("Dataframe Size: ", len(df))
    #print("Number of unique titles: ", len(df['title'].unique()) )
    
    df.to_csv('./Data/Cleaned_data/movies.csv', index = False)
    return


def clean_titles_df(df):
    # drop budget, id, production_countries, crew
    #columns_to_drop = ['budget', 'id', 'production_countries', 'crew']
    
    df = df.dropna()
    
    
    print(df.head())
    print("Dataframe columns: ", df.columns)    
    print("Dataframe Size: ", len(df))
    df.to_csv('./Data/Cleaned_data/netflix_titles.csv', index=False)
    return

movies_df = load_and_view_df('./Data/movies (1).csv')
clean_movies_df(movies_df)

titles_df = load_and_view_df('./Data/netflix_titles.csv')
clean_titles_df(titles_df)


