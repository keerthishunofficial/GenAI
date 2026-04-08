import pandas as pd
import numpy as np

def generate_sample_data(filename="sample_data.csv", num_records=100):
    np.random.seed(42)
    
    data = {
        "transaction_id": [f"TXN-{i:04d}" for i in range(1, num_records + 1)],
        "date": pd.date_range(start="2024-01-01", periods=num_records, freq="D").astype(str),
        "product_id": np.random.choice(["P001", "P002", "P003", "P004"], num_records),
        "price": np.random.uniform(10.0, 100.0, num_records),
        "quantity": np.random.randint(1, 5, num_records)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce anomalies
    # 1. Duplicates
    df = pd.concat([df, df.sample(10, random_state=42)], ignore_index=True)
    
    # 2. Missing prices
    missing_indices = np.random.choice(df.index, 5, replace=False)
    df.loc[missing_indices, "price"] = np.nan
    
    # 3. Bad dates
    bad_date_indices = np.random.choice(df.index, 5, replace=False)
    df.loc[bad_date_indices, "date"] = "invalid_date_format"
    
    # 4. Outlier quantity
    df.loc[np.random.choice(df.index, 2, replace=False), "quantity"] = -5
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with shape {df.shape}")

if __name__ == "__main__":
    generate_sample_data()
