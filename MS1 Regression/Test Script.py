####################################################  IMPORTING  ################################################################

import pandas as pd
import numpy as np
import pickle
import datetime 

from sklearn.metrics import mean_squared_error, r2_score


#################################################### READING MODELS ############################################################

encoding_model = pickle.load(open("Models/Encoding model.pkl", "rb"))

normalization_model = pickle.load(open("Models/Normalization model.pkl", "rb"))

GB_model = pickle.load(open("Models/GB model.pkl", "rb"))

XG_model = pickle.load(open("Models\XGBoost.pkl", "rb"))

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
    data["Popularity"] = data["Popularity"].fillna(54.1179130996608)
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





    X = data.drop(['Popularity'], axis=1)
    Y = data["Popularity"]
    

    


    X['Album Release Date'] = X['Album Release Date'].apply(extract_year)


    X['Album Release Date'] = X['Album Release Date'].fillna(X['Album Release Date'].loc[X['Album Release Date'].notna()].astype(str))

    X.rename(columns={'Album Release Date': 'Release Year'}, inplace=True)


    # Apply categorization function to create a new column
    X['ArtistCount'] = X['Artist Names'].apply(lambda x: categorize_names(eval(x)))

    # Apply categorization function to create a new column
    X['Genre'] = X['Artist(s) Genres'].apply(lambda x: categorize_genres(eval(x)))


    current_year = datetime.datetime.now().year

    # Calculate the age of the song
    X['Song Age'] = current_year - X['Release Year']

    X.drop(['Song','Album','Artist(s) Genres','Spotify Link', 'Song Image','Spotify URI', 'Artist Names'], axis = 1 , inplace = True)


    categorical_columns = [i for i in X.columns if X[i].dtype == "O"]

    for col in categorical_columns:
        X[col] = X[col].apply(lambda x: encoding_model.transform([x])[0] if x in encoding_model.classes_ else -1)




    selected_x = X[['Release Year', 'Hot100 Ranking Year', 'Hot100 Rank', 'Song Length(ms)',
        'Acousticness', 'Danceability', 'Energy', 'Instrumentalness',
        'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Mode',
        'Time Signature', 'ArtistCount', 'Genre', 'Song Age']]

    X_selected = pd.DataFrame(data= selected_x)

    df= normalization_model.transform(X_selected)

    X= pd.DataFrame(df, columns= X_selected.columns)

    


    
    y_pred = GB_model.predict(X)

   
    mse = mean_squared_error(Y, y_pred)
    print("Gradient Boost Model \n")
    print(f" mean squared error (MSE) for train: {mse}")
    rmse = np.sqrt(mse)
    print(f" root mean squared error (RMSE) for train: {rmse}")

    rscore = r2_score(Y, y_pred)
    print(f"R2 Score for test: {rscore} ")




    y_pred = XG_model.predict(X)


    mse = mean_squared_error(Y, y_pred)
    print("XGBoost Model \n")
    print(f" mean squared error (MSE) for train: {mse}")
    rmse = np.sqrt(mse)
    print(f" root mean squared error (RMSE) for train: {rmse}")

    rscore = r2_score(Y, y_pred)
    print(f"R2 Score for test: {rscore} ")




############################################################### USAGE ###################################################################

data_path = "D:\Fcis\Year3\Semester 2\Machine\Project\Song Popularity\MS1 Regression\SongPopularity_reg_test.csv"



Test(data_path)





