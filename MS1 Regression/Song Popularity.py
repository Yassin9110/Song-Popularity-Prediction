import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest , f_regression
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("SongPopularity.csv")


print("Nulls in data: \n", data.isnull().sum())

print("/////////////////////////////////////////////////////")



######################################################################################################################



print("Duplicates in data: ",data.duplicated().sum())

print("/////////////////////////////////////////////////////")

######################################################################################################################

def extract_year(date_str):
    try:
        return pd.to_datetime(date_str, errors= 'coerce').year
    except:
        return None


data['Album Release Date'] = data['Album Release Date'].apply(extract_year)


data['Album Release Date'] = data['Album Release Date'].fillna(data['Album Release Date'].loc[data['Album Release Date'].notna()].astype(str))

data.rename(columns={'Album Release Date': 'Release Year'}, inplace=True)



######################################################################################################################


def categorize_names(names_list):
    num_names = len(names_list)
    if num_names == 1:
        return 'single'
    elif num_names == 2:
        return 'duo'
    else:
        return 'band'

# Apply categorization function to create a new column
data['ArtistCount'] = data['Artist Names'].apply(lambda x: categorize_names(eval(x)))

data.drop("Artist Names", axis= 1, inplace= True)

######################################################################################################################



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
    
# Apply categorization function to create a new column
data['Genre'] = data['Artist(s) Genres'].apply(lambda x: categorize_genres(eval(x)))



######################################################################################################################




current_year = datetime.now().year

# Calculate the age of the song
data['Song Age'] = current_year - data['Release Year']


######################################################################################################################



print("Outliers in data: \n")
outliers_columns = categorical = [i for i in data.columns if data[i].dtypes != "O"]
for col in outliers_columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    num_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
    print(f"{col},: {num_outliers}")


######################################################################################################################




categorical = [i for i in data.columns if data[i].dtypes == "O"]


for col in categorical:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])



######################################################################################################################







corr_matrix = data.corr()
print("Correlation With target: \n")
print(abs(corr_matrix["Popularity"]).sort_values(ascending=False))


######################################################################################################################


X = data.drop(['Popularity'], axis=1)
Y = data["Popularity"]

######################################################################################################################


k= 17
selected_feature= X.columns[SelectKBest(f_regression, k=k).fit(X,Y).get_support()]
print("///////////////////////////////////////////////////////////////////////////////////////")
print("Selected features is: \n")
print(selected_feature)



selected_x = X[['Release Year', 'Hot100 Ranking Year', 'Hot100 Rank', 'Song Length(ms)',
       'Acousticness', 'Danceability', 'Energy', 'Instrumentalness',
       'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Mode',
       'Time Signature', 'ArtistCount', 'Genre', 'Song Age']]

X_selected = pd.DataFrame(data= selected_x)


######################################################################################################################



normalize = MinMaxScaler()
df= normalize.fit_transform(X_selected)
Nor_x= pd.DataFrame(df, columns= X_selected.columns)

######################################################################################################################



x_train ,x_test, y_train ,y_test = train_test_split(Nor_x, Y, test_size=0.3, random_state = 42)


######################################################################################################################


def model_val2(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    ytrainpred = model.predict(x_train)
    y_pred = model.predict(x_test)
    
    msetrain = mean_squared_error(y_train, ytrainpred)
    print(f"{model} mean squared error (MSE) for train: {msetrain} \n")
    rmsetrain = np.sqrt(msetrain)
    print(f"{model} root mean squared error (RMSE) for train: {rmsetrain} \n")


    mse = mean_squared_error(y_test, y_pred)
    print(f"{model} mean squared error (MSE) for test: {mse} \n")
    
    rmse = np.sqrt(mse)
    print(f"{model} root mean squared error (RMSE): {rmse} \n")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"{model} mean absolute error (MAE) for test: {mae} \n")

    rscore = r2_score(y_test, y_pred)
    print(f"{model} R2 Score for test: {rscore} \n")

    print("/////////////////////////////////////////////////////////////////////////////////////// \n")




######################################################################################################################



LR_model = LinearRegression()
model_val2(LR_model,x_train,y_train, x_test, y_test)



######################################################################################################################



XG_model = XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.01)
model_val2(XG_model,x_train,y_train, x_test, y_test)






# Define the parameter grid for XGBoost
param_grid = {
    'learning_rate': [0.01, 0.03, 0.06, 0.09],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

# Initialize XGBoost regressor
xgb_model = XGBRegressor()

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Using negative mean squared error as the scoring metric
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    verbose=2  # Print detailed information
)


grid_search.fit(x_train, y_train)


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


best_model.fit(x_train, y_train)


y_pred_best = best_model.predict(x_test)
mse_best = mean_squared_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

print("Best hyperparameters:", best_params)
print("Final model MSE with best hyperparameters:", mse_best)
print("Final R2 Score:", best_r2)


######################################################################################################################




poly_reg = PolynomialFeatures(degree=2)
x_train_poly = poly_reg.fit_transform(x_train)
x_test_poly = poly_reg.transform(x_test)
lin_reg2 = LinearRegression()
model_val2(lin_reg2,x_train_poly,y_train, x_test_poly, y_test)


######################################################################################################################


svr_model = SVR(kernel= 'poly', degree= 5)

model_val2(svr_model,x_train,y_train, x_test, y_test)

######################################################################################################################



lsvr_model = LinearSVR(loss ='squared_epsilon_insensitive')
model_val2(lsvr_model,x_train,y_train, x_test, y_test)


######################################################################################################################


lasso_model = Lasso(alpha= 0.01)
model_val2(lasso_model ,x_train,y_train, x_test, y_test)

######################################################################################################################


ridge_model = Ridge()
model_val2(ridge_model ,x_train,y_train, x_test, y_test)

######################################################################################################################



gb_model = GradientBoostingRegressor()

model_val2(gb_model ,x_train,y_train, x_test, y_test)



import random


# Define the parameter grid for Gradient Boosting Regressor
param_grid = {
    'learning_rate': [0.01, 0.03, 0.06,0.09],  # Different learning rates to try
    'n_estimators': [100, 200, 300],    # Number of trees in the forest
    'max_depth': [3, 5, 7],             # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]     # Minimum number of samples required to split an internal node
}

num_iterations = 10
best_mse = float('inf')
best_params = {}

for _ in range(num_iterations):
    params = {
        'learning_rate': random.choice(param_grid['learning_rate']),
        'n_estimators': random.choice(param_grid['n_estimators']),
        'max_depth': random.choice(param_grid['max_depth']),
        'min_samples_split': random.choice(param_grid['min_samples_split'])
    }

    model = GradientBoostingRegressor(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if mse < best_mse:
        best_mse = mse
        best_params = params

best_model = GradientBoostingRegressor(**best_params)
best_model.fit(x_train, y_train)
y_pred_best = best_model.predict(x_test)
mse_best = mean_squared_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)
print("Best hyperparameters:", best_params)
print("Final model MSE with best hyperparameters:", mse_best)
print("Final R2 Score : ", best_r2)






model = GradientBoostingRegressor(**best_params)


train_rmse = []
test_rmse = []


for i in range(1, 401):  
    model.fit(x_train[:i], y_train[:i])  
    y_train_pred = model.predict(x_train[:i])
    y_test_pred = model.predict(x_test)
    train_rmse.append(np.sqrt(mean_squared_error(y_train[:i], y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))


# Plotting the RMSE values
plt.figure(figsize=(10, 6))
plt.plot(range(1, 401), train_rmse, label='Train RMSE')
plt.plot(range(1, 401), test_rmse, label='Test RMSE')
plt.xlabel('Number of Iterations')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Training and Testing RMSE vs. Iterations')
plt.legend()
plt.grid(True)
plt.show()



######################################################################################################################
