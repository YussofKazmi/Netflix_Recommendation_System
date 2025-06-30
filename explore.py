import pandas as pd
import os
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from collections import Counter
from scipy.sparse import csr_matrix

def load_and_view_df(url):
    df = pd.read_csv(url)
    df = df[['title', 'genres','vote_average','vote_count','tagline','popularity','director','cast', 'overview', 'keywords']].dropna()
    print(df.head())
    
    print("Dataframe columns: ", df.columns)    
    print("Dataframe Size: ", len(df))
    print("Number of unique titles: ", len(df['title'].unique()) )
    return df




df = load_and_view_df('./Data/Cleaned_data/movies.csv')
#print(df)

# 1) Parse cast into list
df['cast_list'] = df['cast'].apply(lambda s: [name.strip() for name in s.split() if name])

actor_counts = Counter(a for lst in df['cast_list'] for a in lst)
top_500 = set([a for a, _ in actor_counts.most_common(500)])

df['cast_pruned'] = df['cast_list'].apply(
lambda lst: [a if a in top_500 else 'Other' for a in lst]
        )

# multi label encoding using multi-lable binarizer
mlb_genres = MultiLabelBinarizer()
X_genres = mlb_genres.fit_transform(df['genres'].str.split())




mlb_cast = MultiLabelBinarizer()
X_cast = mlb_cast.fit_transform(df['cast_pruned'])

mlb_director = MultiLabelBinarizer()
X_director = mlb_director.fit_transform(df['director'])


#print(X_director)


# TAGLINE
#text freq inv doc freq analyzer for the count of specific words such as in tagline
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_tl = tfidf.fit_transform(df['tagline'].fillna(''))



# Vectorizing the description

tfidf_desc = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,4), # most specific phrases are 1 to 2 word/s, having 3 means 125,000 combos...
    min_df=10 
)

X_desc = tfidf_desc.fit_transform(df['overview'].fillna(''))


# Vectorizing the keywords

tfidf_kw = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2), # most specific phrases are 1 to 2 word/s, having 3 means 125,000 combos...
    min_df=10 
)

X_keywords = tfidf_kw.fit_transform(df['keywords'].fillna(''))











#scale numericla vals
scaler = StandardScaler()
X_num = scaler.fit_transform(df[['popularity', 'vote_average', 'vote_count']])

print(X_num)

#hstoack to store a column wise rep of dfioff arrays of same shape
#csr maxtrix so we can be more memory efficient, since a lot of these numpy arrays have mostly 0 for elements
X_all = hstack(
    [
        csr_matrix(X_genres),
        X_keywords,
        X_cast,
        csr_matrix(X_director),
        X_tl,
        csr_matrix(X_num),
        X_desc
    ], format='csr')




# finally, copute the similarity matrix to then feed into the reccomendation system
simlarity_matrix =  cosine_similarity(X_all, X_all)

def recommend(title, k=10):
    idx = df.index[df['title']==title][0]
    scores = list(enumerate(simlarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recs = [df['title'].iloc[i] for i, _ in scores[1:k+1]]
    return recs


print(recommend('Pacific Rim'))