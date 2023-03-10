from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc, classification_report
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from scipy import stats

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('loan_data_2007_2014.csv', index_col=[0])

categorical_columns = df.select_dtypes(include='object').columns.tolist()
numerical_columns = df.select_dtypes(include='number').columns.tolist()

# Convert date columns into proper format
def convert_date_columns(df, column):
    # Store current day
    today_date = pd.to_datetime('today')
    
    # Convert column to datetime format
    df[column] = pd.to_datetime(df[column], format = "%b-%y")
    
    # Calculate the difference in months and add to a new column
    df['mths_since_' + column] = round(pd.to_numeric((today_date - df[column]) / np.timedelta64(1, 'M')))
    df['mths_since_' + column] = df['mths_since_' + column].apply(lambda x: df['mths_since_' + column].max() if x < 0 else x)
    
    # Drop the original date column
    df.drop(columns=[column], inplace=True)
convert_date_columns(df, 'earliest_cr_line')
convert_date_columns(df, 'issue_d')
convert_date_columns(df, 'last_pymnt_d')
convert_date_columns(df, 'last_credit_pull_d')

# Convert target variable values into 0 and 1
df['target'] = np.where(df.loc[:,'loan_status'].isin(['Charged Off', 'Default',
                                                      'Late (31-120 days)',
                                                      'Does not meet the credit policy. Status:Charged Off']), 0, 1)
# Drop the original 'loan_status' column
df.drop(columns=['loan_status'], inplace=True)

# Drop irrelevant & forward-looking columns
df.drop(columns=['id', 'member_id', 'sub_grade', 'emp_title', 'url', 'desc', 'title', 'zip_code', 'next_pymnt_d', 
                 'recoveries', 'collection_recovery_fee', 'total_rec_prncp', 'total_rec_late_fee'], inplace = True)

# Drop columns with high percentages of null (>80%)
def null_table(df):
    # Count null values and calculate their percentages 
    null_values = df.isnull().sum()
    null_percent = 100 * null_values / len(df)
    
    # Append columns with high percentages of null (>80%)
    drop_null_columns = []
    for null_column, null_count in null_values.iteritems():
        if (100 * null_count / len(df)) > 80:
            drop_null_columns.append(null_column)
    
    # Assign count and percentage null-valued columns to table
    null_table = pd.concat([null_values, null_percent], axis=1)
    null_table = null_table.rename(columns = {0 : 'Null Values', 1 : 'Null Percent'})
    
    # Show data type of columns with null values
    null_table['Data Type'] = df.dtypes
        
    print("There are " + str(null_table.shape[0]) + " columns that have null values.")
    print("There are " + str(len(drop_null_columns)) + " columns with more than 80% of null values: \n", drop_null_columns)
    
    return null_table, drop_null_columns
 null_table, drop_null_columns = null_table(df)

# Split data into 80/20 while keeping the distribution of bad loans in test set same as that in the pre-split dataset
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Separate categorical and numerical columns from dataframe
X_train_cat = X_train.select_dtypes(include='object').copy()
X_train_num = X_train.select_dtypes(include='number').copy()

# Categorical features are selected based on chi-squared test
chi2_check = {}
for column in X_train_cat:
    chi, p, dof, ex = chi2_contingency(pd.crosstab(y_train, X_train_cat[column]))
    chi2_check.setdefault('Feature',[]).append(column)
    chi2_check.setdefault('p-value',[]).append(round(p, 10))

chi2_result = pd.DataFrame(data = chi2_check)

# Categorical columns are selected where p-value < 0.05 or statistically significant association between X and Y
selected_cat_cols = ['initial_list_status', 'verification_status', 'home_ownership', 'grade', 'purpose', 
                     'addr_state', 'pymnt_plan']

# Numerical features are selected based on ANOVA F test
F_statistic, p_values = f_classif(X_train_num.fillna(X_train_num.median()), y_train)
ANOVA_F_table = pd.DataFrame(data = {'Numerical_Feature': X_train_num.columns.values, 
                                     'F-Score': F_statistic, 
                                     'p values': p_values.round(decimals=10)}).sort_values(by=['F-Score'], ascending=False)

# Select temporary numerical features where p-value < 0.05
temp_selected_num_cols = ANOVA_F_table.iloc[:24,0].to_list()

# Drop the temporarily selected numerical columns based on pair-wise correlation matrix result
dropped_num_cols = ['mths_since_last_pymnt_d','mths_since_last_credit_pull_d','mths_since_issue_d',
                    'total_pymnt', 'total_pymnt_inv', 'installment','funded_amnt', 'funded_amnt_inv',
                    'out_prncp_inv',
                    'total_rev_hi_lim']
selected_num_cols = []
for col in temp_selected_num_cols:
    if col not in dropped_num_cols:
        selected_num_cols.append(col)
        
# Drop non-selected columns from X_train
selected_cols = selected_cat_cols + selected_num_cols
X_train = X_train[selected_cols]

# Drop addr_state column
selected_cat_cols.remove('addr_state')
X_train.drop(columns=['addr_state'], inplace=True)
X_test.drop(columns=['addr_state'], inplace=True)

# Drop pymnt_plan column
selected_cat_cols.remove('pymnt_plan')
X_train.drop(columns=['pymnt_plan'], inplace=True)
X_test.drop(columns=['pymnt_plan'], inplace=True)

# Impute the missing values from numerical columns with its median
my_imputer = SimpleImputer(strategy='median')
X_train[selected_num_cols] = my_imputer.fit_transform(X_train[selected_num_cols])
X_test[selected_num_cols] = my_imputer.transform(X_test[selected_num_cols])

# Drop the previous bin columns
dropped_cols = ['total_rec_int','last_pymnt_amnt','revol_bal', 'loan_amnt', 'mths_since_earliest_cr_line', 'total_acc',
                'inq_last_6mths','emp_length']

for col in dropped_cols:
    selected_num_cols.remove(col)
    X_train.drop(columns=[col], inplace=True)
    X_test.drop(columns=[col], inplace=True)

bin_cols = ['total_rec_int_factor','tot_cur_bal_factor','last_pymnt_amnt_factor','out_prncp_factor','revol_bal_factor',
            'annual_inc_factor','dti_factor','loan_amnt_factor','revol_util_factor','mths_since_earliest_cr_line_factor',
            'int_rate_factor','total_acc_factor','inq_last_6mths_factor']
X_train = X_train.drop(columns=bin_cols)

# Apply one-hot encoding to categorical features
def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df

X_train = dummy_creation(X_train, selected_cat_cols)
X_test = dummy_creation(X_test, selected_cat_cols)
X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)

# Custom transformer class to create new categorical dummy features
class WoE_Binning(BaseEstimator, TransformerMixin):
    def __init__(self, X): 
        self.X = X
    def fit(self, X, y = None):
        return self
    def transform(self, X):        
        # Categorical Features
        X_new = X.loc[:, 'initial_list_status:f'].to_frame()
        X_new['initial_list_status:w'] = X.loc[:, 'initial_list_status:w']
        
        X_new['verification_status:Verified'] = X.loc[:, 'verification_status:Verified']
        X_new['verification_status:Source Verified'] = X.loc[:, 'verification_status:Source Verified']
        X_new['verification_status:Not Verified'] = X.loc[:, 'verification_status:Not Verified']
        
        X_new['home_ownership:OTHER'] = X.loc[:, 'home_ownership:OTHER']
        X_new['home_ownership:NONE'] = X.loc[:, 'home_ownership:NONE']
        X_new['home_ownership:RENT'] = X.loc[:, 'home_ownership:RENT']
        X_new['home_ownership:OWN'] = X.loc[:, 'home_ownership:OWN']
        X_new['home_ownership:MORTGAGE'] = X.loc[:, 'home_ownership:MORTGAGE']
        
        X_new['grade:A'] = X.loc[:, 'grade:A']
        X_new['grade:B'] = X.loc[:, 'grade:B']
        X_new['grade:C'] = X.loc[:, 'grade:C']
        X_new['grade:D'] = X.loc[:, 'grade:D']
        X_new['grade:E'] = X.loc[:, 'grade:E']
        X_new['grade:F'] = X.loc[:, 'grade:F']
        X_new['grade:G'] = X.loc[:, 'grade:G']
        
        X_new['purpose:small_business'] = X.loc[:, 'purpose:small_business']
        X_new['purpose:educational'] = X.loc[:, 'purpose:educational']
        X_new['purpose:renewable_energy'] = X.loc[:, 'purpose:renewable_energy']
        X_new['purpose:moving'] = X.loc[:, 'purpose:moving']
        X_new['purpose:house'] = X.loc[:, 'purpose:house']
        X_new['purpose:other'] = X.loc[:, 'purpose:other']
        X_new['purpose:medical'] = X.loc[:, 'purpose:medical']
        X_new['purpose:vacation'] = X.loc[:, 'purpose:vacation']
        X_new['purpose:wedding'] = X.loc[:, 'purpose:wedding']
        X_new['purpose:debt_consolidation'] = X.loc[:, 'purpose:debt_consolidation']
        X_new['purpose:home_improvement'] = X.loc[:, 'purpose:home_improvement']
        X_new['purpose:major_purchase'] = X.loc[:, 'purpose:major_purchase']
        X_new['purpose:car'] = X.loc[:, 'purpose:car']
        X_new['purpose:credit_card'] = X.loc[:, 'purpose:credit_card']
        
        # Numerical features
        X_new['int_rate:Under 7.484'] = np.where((X['int_rate'] < 7.484), 1, 0)
        X_new['int_rate:7.484-9.548'] = np.where((X['int_rate'] > 7.484) & (X['int_rate'] <= 9.548), 1, 0)
        X_new['int_rate:9.548-11.612'] = np.where((X['int_rate'] > 9.548) & (X['int_rate'] <= 11.612), 1, 0)
        X_new['int_rate:11.612-13.676'] = np.where((X['int_rate'] > 11.612) & (X['int_rate'] <= 13.676), 1, 0)
        X_new['int_rate:13.676-15.74'] = np.where((X['int_rate'] > 13.676) & (X['int_rate'] <= 15.74), 1, 0)
        X_new['int_rate:15.74-17.804'] = np.where((X['int_rate'] > 15.74) & (X['int_rate'] <= 17.804), 1, 0)
        X_new['int_rate:17.804-19.868'] = np.where((X['int_rate'] > 17.804) & (X['int_rate'] <= 19.868), 1, 0)
        X_new['int_rate:19.868-21.932'] = np.where((X['int_rate'] > 19.868) & (X['int_rate'] <= 21.932), 1, 0)
        X_new['int_rate:21.932-23.996'] = np.where((X['int_rate'] > 21.932) & (X['int_rate'] <= 23.996), 1, 0)
        X_new['int_rate:Above 23.996'] = np.where((X['int_rate'] > 23.996), 1, 0)
        
        X_new['revol_util:Under 11.49'] = np.where((X['revol_util'] < 11.49), 1, 0)
        X_new['revol_util:11.49-22.98'] = np.where((X['revol_util'] > 11.49) & (X['revol_util'] <= 22.98), 1, 0)
        X_new['revol_util:22.98-34.47'] = np.where((X['revol_util'] > 22.98) & (X['revol_util'] <= 34.47), 1, 0)
        X_new['revol_util:34.47-45.96'] = np.where((X['revol_util'] > 34.47) & (X['revol_util'] <= 45.96), 1, 0)
        X_new['revol_util:45.96-57.45'] = np.where((X['revol_util'] > 45.96) & (X['revol_util'] <= 57.45), 1, 0)
        X_new['revol_util:57.45-68.94'] = np.where((X['revol_util'] > 57.45) & (X['revol_util'] <= 68.94), 1, 0)
        X_new['revol_util:68.94-80.43'] = np.where((X['revol_util'] > 68.94) & (X['revol_util'] <= 80.43), 1, 0)
        X_new['revol_util:80.43-91.92'] = np.where((X['revol_util'] > 80.43) & (X['revol_util'] <= 91.92), 1, 0)
        X_new['revol_util:91.92-103.41'] = np.where((X['revol_util'] > 91.92) & (X['revol_util'] <= 103.41), 1, 0)
        X_new['revol_util:Above 103.41'] = np.where((X['revol_util'] > 103.41), 1, 0)
        
        X_new['dti:Under 3.999'] = np.where((X['dti'] < 3.999), 1, 0)
        X_new['dti:3.999-7.998'] = np.where((X['dti'] > 3.999) & (X['dti'] <= 7.998), 1, 0)
        X_new['dti:7.998-11.997'] = np.where((X['dti'] > 7.998) & (X['dti'] <= 11.997), 1, 0)
        X_new['dti:11.997-15.996'] = np.where((X['dti'] > 11.997) & (X['dti'] <= 15.996), 1, 0)
        X_new['dti:15.996-19.995'] = np.where((X['dti'] > 15.996) & (X['dti'] <= 19.995), 1, 0)
        X_new['dti:19.995-23.994'] = np.where((X['dti'] > 19.995) & (X['dti'] <= 23.994), 1, 0)
        X_new['dti:23.994-27.993'] = np.where((X['dti'] > 23.994) & (X['dti'] <= 27.993), 1, 0)
        X_new['dti:27.993-31.992'] = np.where((X['dti'] > 27.993) & (X['dti'] <= 31.992), 1, 0)
        X_new['dti:31.992-35.991'] = np.where((X['dti'] > 31.992) & (X['dti'] <= 35.991), 1, 0)
        X_new['dti:Above 35.991'] = np.where((X['dti'] > 35.991), 1, 0)
        
        X_new['annual_inc:Under 76706.4'] = np.where((X['annual_inc'] < 76706.4), 1, 0)
        X_new['annual_inc:76706.4-151516.8'] = np.where((X['annual_inc'] > 76706.4) & (X['annual_inc'] <= 151516.8), 1, 0)
        X_new['annual_inc:151516.8-226327.2'] = np.where((X['annual_inc'] > 151516.8) & (X['annual_inc'] <= 226327.2), 1, 0)
        X_new['annual_inc:226327.2-301137.6'] = np.where((X['annual_inc'] > 226327.2) & (X['annual_inc'] <= 301137.6), 1, 0)
        X_new['annual_inc:301137.6-375948.0'] = np.where((X['annual_inc'] > 301137.6) & (X['annual_inc'] <= 375948.0), 1, 0)
        X_new['annual_inc:375948.0-450758.4'] = np.where((X['annual_inc'] > 375948.0) & (X['annual_inc'] <= 450758.4), 1, 0)
        X_new['annual_inc:450758.4-525568.8'] = np.where((X['annual_inc'] > 450758.4) & (X['annual_inc'] <= 525568.8), 1, 0)
        X_new['annual_inc:525568.8-600379.2'] = np.where((X['annual_inc'] > 525568.8) & (X['annual_inc'] <= 600379.2), 1, 0)
        X_new['annual_inc:600379.2-675189.6'] = np.where((X['annual_inc'] > 600379.2) & (X['annual_inc'] <= 675189.6), 1, 0)
        X_new['annual_inc:Above 675189.6'] = np.where((X['annual_inc'] > 675189.6), 1, 0)
        
        X_new['out_prncp:Under 3216.038'] = np.where((X['out_prncp'] < 3216.038), 1, 0)
        X_new['out_prncp:3216.038-6432.076'] = np.where((X['out_prncp'] > 3216.038) & (X['out_prncp'] <= 6432.076), 1, 0)
        X_new['out_prncp:6432.076-9648.114'] = np.where((X['out_prncp'] > 6432.076) & (X['out_prncp'] <= 9648.114), 1, 0)
        X_new['out_prncp:9648.114-12864.152'] = np.where((X['out_prncp'] > 9648.114) & (X['out_prncp'] <= 12864.152), 1, 0)
        X_new['out_prncp:12864.152-16080.19'] = np.where((X['out_prncp'] > 12864.152) & (X['out_prncp'] <= 16080.19), 1, 0)
        X_new['out_prncp:16080.19-19296.228'] = np.where((X['out_prncp'] > 16080.19) & (X['out_prncp'] <= 19296.228), 1, 0)
        X_new['out_prncp:19296.228-22512.266'] = np.where((X['out_prncp'] > 19296.228) & (X['out_prncp'] <= 22512.266), 1, 0)
        X_new['out_prncp:22512.266-25728.304'] = np.where((X['out_prncp'] > 22512.266) & (X['out_prncp'] <= 25728.304), 1, 0)
        X_new['out_prncp:25728.304-28944.342'] = np.where((X['out_prncp'] > 25728.304) & (X['out_prncp'] <= 28944.342), 1, 0)
        X_new['out_prncp:Above 28944.342'] = np.where((X['out_prncp'] > 28944.342), 1, 0)
        
        X_new['tot_cur_bal:Under 41242.9'] = np.where((X['tot_cur_bal'] < 41242.9), 1, 0)
        X_new['tot_cur_bal:41242.9-82485.8'] = np.where((X['tot_cur_bal'] > 41242.9) & (X['tot_cur_bal'] <= 82485.8), 1, 0)
        X_new['tot_cur_bal:82485.8-123728.7'] = np.where((X['tot_cur_bal'] > 82485.8) & (X['tot_cur_bal'] <= 123728.7), 1, 0)
        X_new['tot_cur_bal:123728.7-164971.6'] = np.where((X['tot_cur_bal'] > 123728.7) & (X['tot_cur_bal'] <= 164971.6), 1, 0)
        X_new['tot_cur_bal:164971.6-206214.5'] = np.where((X['tot_cur_bal'] > 164971.6) & (X['tot_cur_bal'] <= 206214.5), 1, 0)
        X_new['tot_cur_bal:206214.5-247457.4'] = np.where((X['tot_cur_bal'] > 206214.5) & (X['tot_cur_bal'] <= 247457.4), 1, 0)
        X_new['tot_cur_bal:247457.4-288700.3'] = np.where((X['tot_cur_bal'] > 247457.4) & (X['tot_cur_bal'] <= 288700.3), 1, 0)
        X_new['tot_cur_bal:288700.3-329943.2'] = np.where((X['tot_cur_bal'] > 288700.3) & (X['tot_cur_bal'] <= 329943.2), 1, 0)
        X_new['tot_cur_bal:329943.2-371186.1'] = np.where((X['tot_cur_bal'] > 329943.2) & (X['tot_cur_bal'] <= 371186.1), 1, 0)
        X_new['tot_cur_bal:Above 371186.1'] = np.where((X['tot_cur_bal'] > 371186.1), 1, 0)
        
        return X_new
      
# Define modelling pipeline
LOGReg = LogisticRegression(max_iter=1000, class_weight='balanced')
woe_transform = WoE_Binning(X)
pipeline = Pipeline(steps=[('woe', woe_transform), ('model', LOGReg)])

# Define cross-validation criteria. 
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# Fit and evaluate the logistic regression pipeline
scores = cross_val_score(pipeline, X_train, y_train, scoring = 'roc_auc', cv=cv)
AUROC = np.mean(scores)
GINI = AUROC * 2 - 1

# Print the mean AUROC score and Gini
print('Mean AUROC: %.4f' % (AUROC))
print('Gini: %.4f' % (GINI))

# Fit the pipeline on the whole training set
pipeline.fit(X_train, y_train)

# Make preditions on test set
y_hat_test = pipeline.predict(X_test)

# Get the predicted probabilities
y_hat_test_proba = pipeline.predict_proba(X_test)

# Create a scorecard
min_score, max_score = 300, 850
df_scorecard = summary_table
df_scorecard['Features'] = df_scorecard['Feature name'].str.split(':').str[0]
df_scorecard['SubFeatures'] = df_scorecard['Feature name'].str.split(':').str[1]

# Calculate the sum of the minimum & maximum coefficients of each category
min_sum_coef = df_scorecard.groupby('Features')['Coefficients'].min().sum()
max_sum_coef = df_scorecard.groupby('Features')['Coefficients'].max().sum()

# Calculate credit score for all observations
X_test_woe_transformed = woe_transform.transform(X_test)
X_test_woe_transformed.insert(0, 'Intercept', 1)

scorecard_scores = df_scorecard['Scores'].to_frame().to_numpy()
y_scores = X_test_woe_transformed.dot(scorecard_scores)
X_test['Scores'] = y_scores.values

