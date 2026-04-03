import pandas as pd
import numpy as np

def get_easy_task() -> pd.DataFrame:
    """Dataset with nulls in 1-2 columns."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25.0, np.nan, 30.0, np.nan, 22.0],
        'score': [85.0, 90.0, np.nan, 88.0, 92.0]
    }
    return pd.DataFrame(data)

def get_medium_task() -> pd.DataFrame:
    """Dataset includes multiple date formats and repeated customer rows."""
    data = {
        'id': [1, 2, 3, 4, 3, 5],
        'date': ['2023-01-01', '01/02/2023', '2023/01/03', 'Jan 4, 2023', '2023/01/03', '2023-01-05'],
        'amount': [100.0, 200.0, 150.0, 300.0, 150.0, 250.0]
    }
    return pd.DataFrame(data)

def get_hard_task() -> pd.DataFrame:
    """Dataset includes nulls, duplicates, invalid categories, typo col names, mixed casing, malformed dates."""
    data = {
        'Cst_ID': [101, 102, 103, 102, 104, 105, 106, 107],
        'First Name': ['john', 'JANE', 'Alice', 'JANE', 'bob ', None, 'EVE', 'alice'],
        'status_cat': ['ACTIVE', 'inactive', 'Pending', 'inactive', 'active', 'ACTIVE', 'UNKNOWN_STATUS', 'ACTIVE'],
        'Signup-Date': ['2023-01-01', '01/02/2023', '2023/01/03', '01/02/2023', 'Jan 4, 2023', '2023-01-05', 'bad_date', None]
    }
    return pd.DataFrame(data)

TASKS = {
    'easy': get_easy_task,
    'medium': get_medium_task,
    'hard': get_hard_task
}
