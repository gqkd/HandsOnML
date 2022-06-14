#%% import and functions

from attr import attributes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zlib import crc32

import hashlib
from sklearn import impute
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import randint
from scipy import stats
#in this implementation if you update the dataset it will generate a totally new train-test
# but if you run twice the code with the same seed the permutation will be the same
def split_train_test(data, test_ratio, seed=10):
    #data in pandas
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#in this implementation hash of the instance is calculated and if is minor of a test_ratio * max_num
# then is placed in the test set

def test_set_check(identifier, test_ratio):
    #calculate the checksum, verify if it is minor of max number of crc32, means
    # that we have to choose test_ratio < 1 and the crc32 bitwise & is for compatibility with python2
    # this is the same
    # return crc32(np.int64(identifier)) < test_ratio * 2**32
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def test_set_check2(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def proportion(data: pd.DataFrame, column: str):
    return data[column].value_counts()/len(data)

#%% exploration
housing = pd.read_csv("housing.csv")
housing.head()
housing.info()

housing["ocean_proximity"].value_counts()
housing.describe()

housing.hist(bins=50,figsize=(20,15))
plt.show()


# %% splitting
#split with sklearn
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#to split we need an identifier for the column
#method 1
housing_with_id = housing.reset_index() #adds index column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

#method 2, create an identifier from lat and long
housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

#categorization of median income
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()

#we want a stratified sample for median income
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
#lets see the proportion in the set
# strat_test_set["income_cat"].value_counts()/len(strat_test_set)
# housing["income_cat"].value_counts()/len(housing)
#i made a function
print(proportion(strat_test_set,"income_cat"))
print(proportion(housing,"income_cat"))

#again for the table
housing_with_id["income_cat"] = pd.cut(housing_with_id["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1,2,3,4,5])
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
compare_prop = pd.DataFrame({
    "Overall":proportion(housing, "income_cat"),
    "Stratified": proportion(strat_test_set, "income_cat"),
    "Random": proportion(test_set, "income_cat"),
}).sort_index()
compare_prop["Rand. %error"] = 100 * compare_prop["Random"] / compare_prop["Overall"] -100
compare_prop["Strat. %error"] = 100 * compare_prop["Stratified"] / compare_prop["Overall"] -100

print(compare_prop)
#remove the income cat
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    

# %% discover data

#let's make a copy to play
housing = strat_train_set.copy()

#scatter plot with the alpha parameter
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()

# %% correlations

#standard correlation coefficient r
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#scatter matrix pandas
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)

#mixing attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",alpha=0.2)


# %% prepare for ML

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#cleaning, total_bedrooms has missing values

housing.dropna(subset=["total_bedrooms"])    #1 get rid of the district
housing.drop("total_bedrooms", axis=1)       #2 get rid of the whole attributeù
median = housing["total_bedrooms"].median()  #3 replace with median
housing["total_bedrooms"].fillna(median, inplace=True)

#alternative with scikit SimpleImputer

imputer = SimpleImputer(strategy='median')

#median can only be computed with numerical value, drop ocean proximity
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

#imputer calculate the medians and store them in statistics_
imputer.statistics_
housing_num.median().values

#replace missing values
X = imputer.transform(housing_num)
#retransforme in pandas
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#dealing with text variables, categorical
housing_cat = housing[["ocean_proximity"]]
housing_cat.head()

#we can replace categorical data with numbers
ordinal_encoder = OrdinalEncoder()
housing_cat_enc = ordinal_encoder.fit_transform(housing_cat)
housing_cat_enc[:10]
ordinal_encoder.categories_

#another way to issue categorical data is to assign them value 0 or 1, 1 if is the
# represented category, ie <1HOCEAN 1 if it is so, 0 in other cases
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_

# %% custom transformer


# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices

housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()

# %% pipelines


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

housing_prepared = full_pipeline.fit_transform(housing)
# %% models

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#lets try on a smaller data set
some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse #too high

tree_reg = DecisionTreeRegressor(random_state=16)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse #too low

#here we can split the training into train and validation set or we can use cross validation
# cross_val_scores wants an utility function (higher better) instead of a cost function (lower better)
# for this reason we compute the opposite of mse and then we square -scores
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_score = np.sqrt(-scores)

print("Scores", tree_rmse_score)
print("Mean", tree_rmse_score.mean())
print("Standard Deviation", tree_rmse_score.std())

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print("Scores", lin_rmse_scores)
print("Mean", lin_rmse_scores.mean())
print("Standard Deviation", lin_rmse_scores.std())
#%% random forest
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse 

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("Scores", forest_rmse_scores)
print("Mean", forest_rmse_scores.mean())
print("Standard Deviation", forest_rmse_scores.std())
#%% svm

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
#you can save models with joblib
# import joblib
# joblib.dump(my_model, "my_model.pkl")
# ...
# my_model_loaded = joblib.load("my_model.pkl")

# %% fine tuning

#grid search for forest regressor

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=16)
#train with 5 folds, (12+6)*5 = 90 rounds for training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
grid_search.best_estimator_

#evaluation scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
pd.DataFrame(grid_search.cv_results_)
   

# %% fine tuning

#randomized search
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=16)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# %% analyze best models and their errors

features_importances = grid_search.best_estimator_.feature_importances_
features_importances

#display importances next to their attributes names
extra_attribs =["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(features_importances, attributes), reverse=True)

# %% evaluate on the test set
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_prediction = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_prediction)
final_rmse = np.sqrt(final_mse)

final_rmse

#compute a 95% confidence interval for rmse

confidence = 0.95
squared_errors = (final_prediction - y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors))))

#we could compute manually the interval
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
print(np.sqrt(mean - tmargin), np.sqrt(mean + tmargin))

#or we could use the z-score
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
print(np.sqrt(mean - zmargin), np.sqrt(mean + zmargin))

# %% full pipeline with preparation and prediction
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)

#%% exercises transformer to select only important attributes

from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices
np.array(attributes)[top_k_feature_indices]
sorted(zip(feature_importances, attributes), reverse=True)[:k]

preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)

housing_prepared_top_k_features[0:3]

housing_prepared[0:3, top_k_feature_indices]

#%% pipeline with full data preparation plus prediction
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])

prepare_select_and_predict_pipeline.fit(housing, housing_labels)

some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))

#%% automatically explore preparations with GridSearchCV
full_pipeline.named_transformers_["cat"].handle_unknown = 'ignore'

param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)

grid_search_prep.best_params_
