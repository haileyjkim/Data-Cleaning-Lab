# Step 1
# College Completion Dataset: Are students in private institutions more likely to graduate on time?
# Campus Recruitment Dataset: Does the field of study influence the likelihood of receiving job offers?

# Step 2
# College Completion Dataset: Are students in private institutions more likely to graduate on time?
# Independent business metric: 100% graduation rate
#%%
# Prep 
# import dependencies:
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#%%
# read in college completion dataset:
college_url = "https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv"
college = pd.read_csv(college_url)
college.info()

#%% 
# correct variable type/class
college['grad_100_percentile'] = pd.to_numeric(college['grad_100_percentile'], errors='coerce')

# collapse factor levels: changing 'control' variable to just 'Private' and 'Public'
college['control'] = college['control'].apply(lambda x: 'Private' if 'Private' in x else 'Public')

# create a target variable: 1 if the institution has a graduation rate above the median, 0 otherwise
median_grad = college['grad_100_percentile'].median()
college['high_grad_rate'] = (college['grad_100_percentile'] > median_grad).astype(int)

#%%
# drop unneeded variables:
relevant_cols = ['control', 'level', 'student_count', 'high_grad_rate']
college = college[relevant_cols]

# one-hot encoding factor variables 
college = pd.get_dummies(college, columns=['control', 'level'], drop_first=True)

# normalize 
college[['student_count']] = MinMaxScaler().fit_transform(college[['student_count']])

# prevalence
prevalence = college['high_grad_rate'].mean()
print(f"Prevalence of High Success: {prevalence:.2%}")

#%%
# partitions 
# first split
train, test = train_test_split(
    college,
    train_size=0.6, 
    stratify=college.high_grad_rate,
    random_state=42
)
# Verify the split sizes
print(f"Training set shape: {train.shape}")
print(f"Remaining (Tune/Test) set shape: {test.shape}")

#%%
# second split 
tune, test = train_test_split(
    test,
    train_size=0.5,
    stratify=test.high_grad_rate,
    random_state=42
)

# Verify final sizes
print(f"Tuning set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")

#%%
# Step 3
# The data is sufficient to answer my question, though comparing private vs. public
# instiitutions may be confounded by other factors like money, prestige, etc. 

#%%
# step 2
# Campus Recruitment Dataset: Does the field of study influence the likelihood of receiving job offers?
# Independent business metric: Job Offer Rate

# read in the campus recruitment dataset:
campus_url = 'https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv'
campus = pd.read_csv(campus_url)
campus.info()

# %%
# correct variable type/class
campus['degree_p'] = pd.to_numeric(campus['degree_p'], errors='coerce')

# collapse factor levels: collapse 'degree_t' categories into 'Mainstream' (Sci&Tech/Comm&Mgmt) and 'Others'
campus['degree_type'] = campus['degree_t'].apply(lambda x: 'Mainstream' if x in ['Sci&Tech', 'Comm&Mgmt'] else 'Others')

# create a target variable
campus['job_offer'] = (campus['status'] == 'Placed').astype(int)

#%%
# drop unneeded variables:
relevant_cols = ['degree_type', 'degree_p', 'job_offer']
campus = campus[relevant_cols]

# one-hot encoding factor variables 
campus = pd.get_dummies(campus, columns=['degree_type'], drop_first=True)

# normalize
campus[['degree_p']] = MinMaxScaler().fit_transform(campus[['degree_p']])

# prevalence
prevalence = campus['job_offer'].mean()
print(f"Prevalence of Job Offers: {prevalence:.2%}")

#%% 
# partitions
# first split
train, test = train_test_split(
    campus,
    train_size=0.6, 
    stratify=campus.job_offer,
    random_state=42
)
# Verify the split sizes
print(f"Training set shape: {train.shape}")

#%%
# second split
tune, test = train_test_split(
    test,
    train_size=0.5,
    stratify=test.job_offer,
    random_state=42
)

# Verify final sizes
print(f"Tuning set shape: {tune.shape}")
print(f"Test set shape: {test.shape}")

# %%
# step 3
# The data is sufficient to answer my question, though comparing the field of study
# may be confounded by other factors like prior work experience, academic history, and salary trade-offs.