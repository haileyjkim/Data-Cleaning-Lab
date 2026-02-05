import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# College Completion Dataset Pipeline
def college_pipeline(url):
    # Load
    df = pd.read_csv(url)
    
    # 1. Correct variable type/class
    df['grad_100_percentile'] = pd.to_numeric(df['grad_100_percentile'], errors='coerce')

    # 2. Collapse factor levels
    df['control'] = df['control'].apply(lambda x: 'Private' if 'Private' in str(x) else 'Public')

    # 3. Create target variable
    median_grad = df['grad_100_percentile'].median()
    df['high_grad_rate'] = (df['grad_100_percentile'] > median_grad).astype(int)

    # 4 & 5. Drop unneeded variables and One-hot encoding
    relevant_cols = ['control', 'level', 'student_count', 'high_grad_rate']
    df = df[relevant_cols].dropna()
    df = pd.get_dummies(df, columns=['control', 'level'], drop_first=True)

    # 6. Normalize
    df[['student_count']] = MinMaxScaler().fit_transform(df[['student_count']])

    # 7. Prevalence
    print(f"College Success Prevalence: {df['high_grad_rate'].mean():.2%}")

    # 8. Partitions (60/20/20 split)
    train, rest = train_test_split(df, train_size=0.6, stratify=df.high_grad_rate, random_state=42)
    tune, test = train_test_split(rest, train_size=0.5, stratify=rest.high_grad_rate, random_state=42)
    
    print(f"College Partition Shapes: Train {train.shape}, Tune {tune.shape}, Test {test.shape}")
    return train, tune, test


# Does the field of study influence the likelihood of receiving job offers?
def campus_pipeline(url):
    # Load
    df = pd.read_csv(url)
    
    # 1. Correct variable type/class
    df['degree_p'] = pd.to_numeric(df['degree_p'], errors='coerce')

    # 2. Collapse factor levels
    df['degree_type'] = df['degree_t'].apply(lambda x: 'Mainstream' if x in ['Sci&Tech', 'Comm&Mgmt'] else 'Others')

    # 3. Create target variable
    df['job_offer'] = (df['status'] == 'Placed').astype(int)

    # 4 & 5. Drop unneeded variables and One-hot encoding
    relevant_cols = ['degree_type', 'degree_p', 'job_offer']
    df = df[relevant_cols].dropna()
    df = pd.get_dummies(df, columns=['degree_type'], drop_first=True)

    # 6. Normalize
    df[['degree_p']] = MinMaxScaler().fit_transform(df[['degree_p']])

    # 7. Prevalence
    print(f"Placement Prevalence: {df['job_offer'].mean():.2%}")

    # 8. Partitions (60/20/20 split)
    train, rest = train_test_split(df, train_size=0.6, stratify=df.job_offer, random_state=42)
    tune, test = train_test_split(rest, train_size=0.5, stratify=rest.job_offer, random_state=42)
    
    print(f"Campus Partition Shapes: Train {train.shape}, Tune {tune.shape}, Test {test.shape}")
    return train, tune, test