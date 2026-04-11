import pandas as pd
import numpy as np

def get_easy_task() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25.0, np.nan, 30.0, np.nan, 22.0],
        "score": [85.0, 90.0, np.nan, 88.0, 92.0],
    }
    return pd.DataFrame(data)

def get_medium_task() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 3, 5],
        "date": [
            "2023-01-01",
            "01/02/2023",
            "2023/01/03",
            "Jan 4, 2023",
            "2023/01/03",
            "2023-01-05",
        ],
        "amount": [100.0, 200.0, 150.0, 300.0, 150.0, 250.0],
    }
    return pd.DataFrame(data)

def get_hard_task() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5, 2, 6],
        "customer_name": ["Acme Corp", "Beta LLC", " Gamma Inc ", "Delta-Co", "Epsilon", "Beta LLC", "Zeta_corp"],
        "contact_date": ["2023-01-01", "2023-01-05", "15-01-2023", "2023/01/20", "Jan 25 2023", "2023-01-05", "2023-02-01"],
        "revenue": [50000.0, 75000.0, np.nan, 120000.0, -100.0, 75000.0, 90000.0],
        "status": ["ACTIVE", "ACTIVE", "UNKNOWN_STATUS", "ACTIVE", "INACTIVE", "ACTIVE", "UNKNOWN_STATUS"]
    }
    return pd.DataFrame(data)

TASKS = {
    "easy": get_easy_task,
    "medium": get_medium_task,
    "hard": get_hard_task,
}