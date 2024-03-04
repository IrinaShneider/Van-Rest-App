

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.title('Vancouver Restaurant Recommender')

# Lets download the data
df_rest = pd.read_csv('C:\Users\ishn3001\app\df_for_app_3')



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = "english", min_df=2)
df_rest['review'] = df_rest['review'].fillna("")

TF_IDF_matrix = vectorizer.fit_transform(df_rest['review'])



from sklearn.metrics.pairwise import cosine_similarity 
similarities = cosine_similarity(TF_IDF_matrix, dense_output=False)




def content_recommender(restaurant, similarities, review_threshold) :
    
    # Get the restaurant by the title
    restaurant_index = df_rest[df_rest['restaurant'] == restaurant].index
    
    # Create a dataframe with the restautant names
    sim_df = pd.DataFrame(
        {'restaurant': df_rest['restaurant'], 
         'similarity': np.array(similarities[restaurant_index, :].todense()).squeeze(),
         'review_count': df_rest['review_count'],
         'url': df_rest['url']
        })
    
    # Filter restaurants with more than the specified review threshold
    sim_df = sim_df[sim_df['review_count'] > review_threshold]
    
    
    # Get the top 10 movies with review threshold > 100 review
    top_restaurants = sim_df.sort_values(by='similarity', ascending=False)

    
    return top_restaurants[1:num_recommendations + 1]

# Sidebar
st.sidebar.title("Restaurant Name")
restaurant= st.sidebar.selectbox("Select a restaurant:", df_rest['restaurant'])
num_recommendations = st.sidebar.slider("Number of similar restaurants to recommend:", 1, 10, 5)


# Display similar restaurants
st.write(f"Top {num_recommendations} restaurants similar to '{restaurant}':")
similar_rest = content_recommender(restaurant, similarities, num_recommendations)
# Display the similar restaurants as clickable links
for index, row in similar_rest.iterrows():
    st.markdown(f"[{row['restaurant']}]({row['url']})")

