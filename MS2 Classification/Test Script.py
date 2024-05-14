####################################################  IMPORTING  ################################################################

import pandas as pd
import numpy as np
import pickle
import datetime 
from sklearn.metrics import accuracy_score




#################################################### READING MODELS ############################################################

encoding_model = pickle.load(open("Models/Encoding model.pkl", "rb"))

normalization_model = pickle.load(open("Models/Normalization model.pkl", "rb"))

GB_model = pickle.load(open("Models/GB model.pkl", "rb"))

voting_model = pickle.load(open("Models/voting model.pkl", "rb"))

stacking_model = pickle.load(open("Models/stacking model.pkl", "rb"))

####################################################  PREPROCESSING FUNCTIONS #################################################

def extract_year(date_str):
    try:
        return pd.to_datetime(date_str, errors= 'coerce').year
    except:
        return None

def categorize_names(names_list):
    num_names = len(names_list)
    if num_names == 1:
        return 'single'
    elif num_names == 2:
        return 'duo'
    else:
        return 'band'

def categorize_genres(genres_list):
    if any('pop' in genre.lower() for genre in genres_list):
        return 'Pop'
    elif any('hip hop' in genre.lower() for genre in genres_list):
        return 'Hip Hop/Rap'
    elif any('country' in genre.lower() for genre in genres_list):
        return 'Country'
    elif any('electropop' in genre.lower() for genre in genres_list):
        return 'Electropop'
    elif any('standards' in genre.lower() for genre in genres_list):
        return 'Adult Standards'
    else:
        return 'Others'


############################################# TEST FUCNCTION ################################################



def Test(datapath):

    data = pd.read_csv(datapath)


    data["Song"] = data["Song"].fillna("Heaven")
    data["Album"] = data["Album"].fillna("Greatest Hits")
    data["Album Release Date"] = data["Album Release Date"].fillna("1/1/2010")
    data["Artist Names"] = data["Artist Names"].fillna("The Karaoke Channel")
    data["Artist(s) Genres"] = data["Artist(s) Genres"].fillna("")
    data["Spotify Link"] = data["Spotify Link"].fillna("https://open.spotify.com/track/0bYg9bo50gSsH3LtXe2SQn")
    data["Song Image"] = data["Song Image"].fillna("https://i.scdn.co/image/ab67616d00001e021fc9fd5d701ee05cb39b7b19")
    data["Spotify URI"] = data["Spotify URI"].fillna("spotify:track:0bYg9bo50gSsH3LtXe2SQn")
    data["PopularityLevel"] = data["PopularityLevel"].fillna("Average")
    data["Hot100 Ranking Year"] = data["Hot100 Ranking Year"].fillna(1987.8481666935875)
    data["Hot100 Rank"] = data["Hot100 Rank"].fillna(48.320626716200934)
    data["Song Length(ms)"] = data["Song Length(ms)"].fillna(224626.53529316752)
    data["Acousticness"] = data["Acousticness"].fillna(0.3000630259602649)
    data["Danceability"] = data["Danceability"].fillna(0.6177100629946697)
    data["Energy"] = data["Energy"].fillna(0.5968806493296721)
    data["Instrumentalness"] = data["Instrumentalness"].fillna(0.04599186652560168)
    data["Liveness"] = data["Liveness"].fillna(0.18075475690518497)
    data["Loudness"] = data["Loudness"].fillna(-8.718823937974479)
    data["Speechiness"] = data["Speechiness"].fillna(0.07203839444354708)
    data["Tempo"] = data["Tempo"].fillna(119.00919770634792)
    data["Valence"] = data["Valence"].fillna(0.5963210628331449)
    data["Key"] = data["Key"].fillna(5.240833467937328)
    data["Mode"] = data["Mode"].fillna(0.7058633500242287)
    data["Time Signature"] = data["Time Signature"].fillna(3.9410434501696012)





    X = data.drop(['PopularityLevel'], axis=1)
    Y = data['PopularityLevel']

    Y = Y.replace({'Not Popular': 0, 'Average': 1, 'Popular': 2})

    


    X['Album Release Date'] = X['Album Release Date'].apply(extract_year)


    X['Album Release Date'] = X['Album Release Date'].fillna(X['Album Release Date'].loc[X['Album Release Date'].notna()].astype(str))

    X.rename(columns={'Album Release Date': 'Album Release Year'}, inplace=True)


    # Apply categorization function to create a new column
    X['ArtistCount'] = X['Artist Names'].apply(lambda x: categorize_names(eval(x)))

    # Apply categorization function to create a new column
    X['Genre'] = X['Artist(s) Genres'].apply(lambda x: categorize_genres(eval(x)))


    current_year = datetime.datetime.now().year

    # Calculate the age of the song
    X['Song Age'] = current_year - X['Album Release Year']

    X.drop(['Song','Album','Artist(s) Genres','Spotify Link', 'Song Image','Spotify URI', 'Artist Names'], axis = 1 , inplace = True)


    categorical_columns = [i for i in X.columns if X[i].dtype == "O"]

    for col in categorical_columns:
        X[col] = X[col].apply(lambda x: encoding_model.transform([x])[0] if x in encoding_model.classes_ else -1)




    x_numerical = X.drop(['Key','Mode','Time Signature','ArtistCount','Genre'],axis= 1)

    x_normalized = normalization_model.transform(x_numerical)
    x_normalized= pd.DataFrame(x_normalized, columns= x_numerical.columns)
    x_normalized.reset_index(drop=True, inplace=True)


    X.reset_index(drop=True, inplace=True)



    columns_to_replace = ['Hot100 Ranking Year', 'Album Release Year', 'Song Age', 'Loudness', 'Acousticness', 'Energy', 'Song Length(ms)']


    X[columns_to_replace] = x_normalized[columns_to_replace]


    X = X[['Hot100 Ranking Year', 'Album Release Year', 'Song Age', 'Loudness', 'Acousticness', 'Energy', 'Song Length(ms)', 'Key', 'Mode', 'Genre']]

    y_pred = GB_model.predict(X)

    accuracy = accuracy_score(Y, y_pred)
    print(f"accuracy for Gradient boost model =  {accuracy}")


    y_pred = voting_model.predict(X)

    accuracy = accuracy_score(Y, y_pred)
    print(f"accuracy for Voting model {accuracy}")


    y_pred = stacking_model.predict(X)

    accuracy = accuracy_score(Y, y_pred)
    print(f"accuracy for Stacking model {accuracy}")





############################################################### USAGE ###################################################################

data_path = "D:\Fcis\Year3\Semester 2\Machine\Project\Song Popularity\MS2 Classification\SongPopularity_classification_test.csv"



Test(data_path)





