# Kaggle-Predicting-Future-Sales

[![Makes people smile](https://forthebadge.com/images/badges/makes-people-smile.svg)](https://github.com/iamsivab)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fiamsivab%2FKaggle-Predicting-Future-Sales)](https://hits.seeyoufarm.com)

## Kaggle-Predicting-Future-Sales

[![Generic badge](https://img.shields.io/badge/Datascience-Beginners-Red.svg?style=for-the-badge)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales) 
[![Generic badge](https://img.shields.io/badge/LinkedIn-Connect-blue.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/iamsivab/) [![Generic badge](https://img.shields.io/badge/Python-Language-blue.svg?style=for-the-badge)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales) [![ForTheBadge uses-git](http://ForTheBadge.com/images/badges/uses-git.svg)](https://GitHub.com/)

#### The goal of this project is to Predict the Future Sales [#DataScience](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales) for the challenging time-series dataset consisting of daily sales data,

[![GitHub repo size](https://img.shields.io/github/repo-size/iamsivab/Kaggle-Predicting-Future-Sales.svg?logo=github&style=social)](https://github.com/iamsivab) [![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/iamsivab/Kaggle-Predicting-Future-Sales.svg?logo=git&style=social)](https://github.com/iamsivab/)[![GitHub top language](https://img.shields.io/github/languages/top/iamsivab/Kaggle-Predicting-Future-Sales.svg?logo=python&style=social)](https://github.com/iamsivab)

#### Few popular hashtags - 
### `#Sales Prediction` `#Time Series` `#Ensembling`
### `#XGBoost` `#Parameter Tuning` `#LightGBM`

### Motivation
In this competition I was working with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. 
To predict total sales for every product and store in the next month. By solving this competition I was able to apply and enhance your data science skills.

This documentation contains general information about my approach and technical information about Kaggle’s Predict Future Sales competition

### Steps involved in this project
### Kaggle Predicting Future Sales- Playground Prediction Competition

### Kaggle Competition: [Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)
### Data Description

You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

**File descriptions**
```
- sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
- test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
- sample_submission.csv - a sample submission file in the correct format.
- items.csv - supplemental information about the items/products.
- item_categories.csv  - supplemental information about the items categories.
- shops.csv- supplemental information about the shops.
```

**Data fields**
```
- ID - an Id that represents a (Shop, Item) tuple within the test set
- shop_id - unique identifier of a shop
- item_id - unique identifier of a product
- item_category_id - unique identifier of item category
- item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
- item_price - current price of an item
- date - date in format dd/mm/yyyy
- date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
- item_name - name of item
- shop_name - name of shop
- item_category_name - name of item category
```
## I. Summary
- Main methods I used for this competition that provides the desired Leaderboard score: LightGBM
- Methods I tried to implement but resulted in worse RMSE: XGBoos, Stacking (both simple averaging and metal models such as Linear Regression and shallow random forest)
- The most important features are lag features of previous months, especially the ‘item_cnt_day’ lag features. Some of them, which can be found in my lag dataset, are 
  - **target_lag_1,target_lag_2**: item_cnt_day of each shop – item pair of previous month and previous two months
  - **item_block_target_mean_lag_1, item_block_target_sum_lag_1**: sum and mean of item_cnt_day per item of previous month
Important features are measured from LightGBM model
- Tools I used in this competition are: numpy, pandas, sklearn, XGBoost GPU, LightGBM (running Pytorch)
- All models are tuned on a linux server with Intel i5 processor, 16GB RAM, NVIDIA 1080 GPU. Tuning models took about 8 to 10 hours, and training on the whole dataset took <=5 minutes

[![Made with Python](https://forthebadge.com/images/badges/made-with-python.svg)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales) [![Made with love](https://forthebadge.com/images/badges/built-with-love.svg)](https://www.linkedin.com/in/iamsivab/) [![ForTheBadge built-with-swag](http://ForTheBadge.com/images/badges/built-with-swag.svg)](https://www.linkedin.com/in/iamsivab/)

## II. Exploratory Data Analysis
More information can be found in [EDA notebook](EDA.ipynb)

Basic data analysis is done, including plotting sum and mean of item_cnt_day for each month to find some patterns, exploring missing values, inspecting test set …

Here are few things interesting I found from doing EDA:
- Number of sold items declines over the year
- There are peaks in November and similar item count zic-zac behaviors in June-July-August. This inspires me to look up Russia national holiday and create a Boolean holiday features. More information can be found in ‘Feature Engineering’ section
- Data has no missing values
- Some interesting information from test set analysis:
  - Not all shop_id in training set are used in test set. Test set excludes following shops (but not vice versa): [0, 1, 8, 9, 11, 13, 17, 20, 23, 27, 29, 30, 32, 33, 40, 43, 51, 54]
  - Not all item in train set are in test set and vice versa
  - In test set, a fixed set of items (5100) are used for each shop_id, and each item only appears one per each shop. This possibly means that items are picked from a generator, which will result in lots of 0 for item count. Therefore, generating all possible shop-item pairs for each month in train set and assigning missing item count with 0 makes sense.


## III. Feature Engineering

### 1. Generate all shop-item pairs and Mean Encoding
Since the competition task is to make a monthly prediction, we need to aggregate the data to monthly level before doing any encodings

Item counts for each shop-item pairs per month (‘target’). I also generated sum and mean of item counts for each shop per month (‘shop_block_target_sum’,’shop_block_target_mean’), each item per month (‘item_block_target_sum’,’item_block_target_mean’, and each item category per month (‘item_cat_block_target_sum’,’item_cat_block_target_mean’)

This process can be found in [this notebook](generate_lag_features.ipynb), under ‘Generating new_sales.csv’. Datasets generated from this steps will be saved under the name ‘new_sales.csv’

### 2. Generate lag features
Lag features are values at prior time steps. I am generating lag features based on ‘item_cnt’ and grouped by ‘shop_id’ and ‘item_id’ .  Time steps are: 1,2,3,5 and 12 months.

All sale record before 2014 are dropped, since there would be no lag features before 2014 as we have a 12-month lag.

These lag features turn out to be the most important features in my dataset, based on gradient boosting’s importance features.

More information can be found in [this notebook](generate_lag_features.ipynb), under ‘Generate lag feature new_sales_lag_after12.pickle’

### 3. Holiday Boolean features
As mentioned above, I look up few Russia national holidays and created few 5 more features: December (to mark December), Newyear_Xmas (for January), Valentine_Menday (February), Women_Day (March), Easter_Labor (April). This might help boosting my score a little since December feature seems to be helpful

After all this steps, you should have a pickle file name in ‘data‘ directory: 'new_sales_lag_after12.pickle'. This is the main file I used for training models


### IV. Cross validations
Since this is time series so I have to pre-define which data can be used for train and test. I have a function called get_cv_idxs in utils.py that will return a list of tuples for cross validation. I decide to use 6 folds, from date_block_num 28 to 33, and luckily this CV score is consistent to leaderboard score.

CV indices can be retrieved from this custom function:

```
cv = get_cv_idxs(dataframe,28,33) 
# dataframe must contain date_block_num features
```

Results from this function can be passed to sklearn GridSearchCV.

### V. Training methods:

### 1. LightGBM
LightGBM is tuned using hyperopt, then manually tune with GridSearchCV to get the optimal result. One interesting thing I found: when tuning the size of the tree, it’s better to tune min_data_in_leaf instead of max_depth. This means to let the tree grows freely until the condition for min_data_in_leaf is met. I believe this will allow deeper logic to develop without overfitting too much. Colsample_bytree and subsample are also used to control overfitting. And I keep the learning rate small (0.03) throughout tuning.

Mean RMSE of 6 folds CV is 0.8088, which is better than any other models I used.

You can find more information in [LGB notebook](lightgbm_tuning.ipynb). From this file I also created out-of-fold features for block 29 to 33, which is used for ensembling later.

Also from this notebook, you can get the leaderboard submission under the file name: ‘coursera_tuned_lightgbm_basic_6folds.csv'

(Note: I do not include some of hyper parameter tuning results from hyperopt since I tuned it at work and I do not have access to that machine now)


### 2. XGBoost
I ran the XGBoost with GPU version, and I follow the same tuning procedures as mentioned in LightGBM. For some reason, I can’t seem to get a consistent result while running XGBoost, even with the same parameters. One example is I get .812 CV score from hyperopt, but I can’t seem to get that result again when getting out-of-fold features (it jumps to .817). This never happens while using LightGBM.

Therefore, I pick 2 models: one with max_depth tuned, and one without max_depth tuned, to get out-of-fold features and hoping they are different enough for ensembling. 

For the record, the first models results .812 CV score (in hyperopt) and .926 LB score, and second models results in .813 CV score (hyperopt) and .927 LB score. Either way, both are worse than LGB model 

``` python 
space = {
    #'n_estimators': hp.quniform('n_estimators', 50, 500, 5),
#     'max_depth': hp.choice('max_depth', np.arange(5, 10, dtype=int)),
    'subsample': hp.quniform('subsample', 0.7, 0.9, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 0.9, 0.05),
    'gamma': hp.quniform('gamma', 0, 1, 0.05),
    'max_leaf_nodes': hp.choice('max_leaf_nodes', np.arange(100,140, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(100,140, dtype=int)),
    'learning_rate': 0.03,
    'eval_metric': 'rmse',
    'objective': 'reg:linear' , 
    'seed': 1204,'tree_method':'gpu_hist'
}

```
best_hyperparams = optimize(space,max_evals=200)
print("The best hyperparameters are: ")
print(best_hyperparams)

You can find more information about this in [XGB notebook](xgb_tuning.ipynb). Prediction for the model with max_depth tuned are named ‘tuned_xgb_basicfeatures_6folds_8126.csv’ and the other one are ‘tuned_xgb_basicfeatures_6folds_8136’


## VI. Ensembling

With LightGBM, XGB model-1 and XGB model-2 out-of-fold features from previous methods, I calculated pairwise differences between them, get the mean of all 3 LGB, XGB1 and XGB2 out-of-fold features, and include the most important features from feature importance: ‘target_lag_1’.

From here I try few ensembling methods
- Simple average and Weighted average 
- SKlearn linear regression and Elasticnet
- Shallow Random Forest, tuned with 5 folds (from 29 to 33)

All of them results in RMSE score that is slightly more than the LightGBM best model, so LightGBM still outperforms them.

``` python
X,y = get_X_y_ensembling(all_oof_df)
params={'alpha': 0.0, 'fit_intercept': False, 'solver': 'sag','random_state':1402}
lr = Ridge(**params)
lr.fit(X,y)
test_pred =  lr.predict(test_df)
pd.Series(test_pred).describe()
get_submission(test_pred,'ensembling_ridge');
```

More information can be found in [Ensembling notebook](ensembling.ipynb)

## VII. Improvement:

Few things that can be improved are:
- Implement neural net WITHOUT categorical embedding
- Generate more feature related to holiday, such as: differences between current month and holiday month.
- Translate item name to English and perform sentiment analysis on item name
- Use only subset of those meta features for ensembling


### Libraries Used

![Ipynb](https://img.shields.io/badge/Python-datetime-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-pandas-blue.svg?style=flat&logo=python&logoColor=white)
![Ipynb](https://img.shields.io/badge/Python-numpy-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-matplotlib-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-seaborn-blue.svg?style=flat&logo=python&logoColor=white)
![Ipynb](https://img.shields.io/badge/Python-scipy-blue.svg?style=flat&logo=python&logoColor=white) 
![Ipynb](https://img.shields.io/badge/Python-sklearn-blue.svg?style=flat&logo=python&logoColor=white) 


### Installation

- Install **datetime** using pip command: `from datetime import datetime`
- Install **pandas** using pip command: `import pandas as pd`
- Install **numpy** using pip command: `import numpy as np`
- Install **matplotlib** using pip command: `import matplotlib`
- Install **matplotlib.pyplot** using pip command: `import matplotlib.pyplot as plt`
- Install **seaborn** using pip command: `import seaborn as sns`
- Install **os** using pip command: `import os`
- Install **scipy** using pip command: `from scipy import sparse`
- Install **scipy.sparse** using pip command: `from scipy.sparse import csr_matrix`
- Install **sklearn.decomposition** using pip command: `from sklearn.decomposition import TruncatedSVD`
- Install **sklearn.metrics.pairwise** using pip command: `from sklearn.metrics.pairwise import cosine_similarity`
- Install **itertools** using pip command: `from itertools import product`


### How to run?

[![Ipynb](https://img.shields.io/badge/Prediction-Sales.Python-lightgrey.svg?logo=python&style=social)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales)


### Project Reports

[![report](https://img.shields.io/static/v1.svg?label=Project&message=Report&logo=microsoft-word&style=social)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/)

- [Download](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/') for the report.

 
### Related Work

[![Sales Prediction](https://img.shields.io/static/v1.svg?label=Sales&message=Prediction&color=lightgray&logo=python&style=social&colorA=critical)](https://www.linkedin.com/in/iamsivab/) [![GitHub top language](https://img.shields.io/github/languages/top/iamsivab/Kaggle-Predicting-Future-Sales.svg?logo=php&style=social)](https://github.com/iamsivab/)

[Sales Prediction](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales) - A Detailed Report on the Analysis


### Contributing

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?logo=github)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/pulls) [![GitHub issues](https://img.shields.io/github/issues/iamsivab/Kaggle-Predicting-Future-Sales?logo=github)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/issues) ![GitHub pull requests](https://img.shields.io/github/issues-pr/viamsivab/Kaggle-Predicting-Future-Sales?color=blue&logo=github) 
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/iamsivab/Kaggle-Predicting-Future-Sales?logo=github)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/)

- Clone [this](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/) repository: 

```bash
git clone https://github.com/iamsivab/Kaggle-Predicting-Future-Sales.git
```

- Check out any issue from [here](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/issues).

- Make changes and send [Pull Request](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/pull).
 
### Need help?

[![Facebook](https://img.shields.io/static/v1.svg?label=follow&message=@iamsivab&color=9cf&logo=facebook&style=flat&logoColor=white&colorA=informational)](https://www.facebook.com/iamsivab)  [![Instagram](https://img.shields.io/static/v1.svg?label=follow&message=@iamsivab&color=grey&logo=instagram&style=flat&logoColor=white&colorA=critical)](https://www.instagram.com/iamsivab/) [![LinkedIn](https://img.shields.io/static/v1.svg?label=connect&message=@iamsivab&color=success&logo=linkedin&style=flat&logoColor=white&colorA=blue)](https://www.linkedin.com/in/iamsivab/)

:email: Feel free to contact me @ [balasiva001@gmail.com](https://mail.google.com/mail/)

[![GMAIL](https://img.shields.io/static/v1.svg?label=send&message=balasiva001@gmail.com&color=red&logo=gmail&style=social)](https://www.github.com/iamsivab) [![Twitter Follow](https://img.shields.io/twitter/follow/iamsivab?style=social)](https://twitter.com/iamsivab)


### License

MIT &copy; [Sivasubramanian](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/blob/master/LICENSE)

[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/0)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/0)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/1)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/1)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/2)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/2)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/3)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/3)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/4)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/4)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/5)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/5)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/6)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/6)[![](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/images/7)](https://sourcerer.io/fame/iamsivab/iamsivab/Kaggle-Predicting-Future-Sales/links/7)


[![GitHub license](https://img.shields.io/github/license/iamsivab/Kaggle-Predicting-Future-Sales.svg?style=social&logo=github)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/blob/master/LICENSE) 
[![GitHub forks](https://img.shields.io/github/forks/iamsivab/Kaggle-Predicting-Future-Sales.svg?style=social)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/network) [![GitHub stars](https://img.shields.io/github/stars/iamsivab/Kaggle-Predicting-Future-Sales.svg?style=social)](https://github.com/iamsivab/Kaggle-Predicting-Future-Sales/stargazers) [![GitHub followers](https://img.shields.io/github/followers/iamsivab.svg?label=Follow&style=social)](https://github.com/iamsivab/)
