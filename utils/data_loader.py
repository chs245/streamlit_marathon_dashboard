import pandas as pd
import json
import os
import streamlit as st

@st.cache_data
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    # Load data types
    dtype_file = os.path.join(base_dir, '../data/ams_marathon2024_dtypes.json')
    if not os.path.isfile(dtype_file):
        st.error(f"Data types file not found at {dtype_file}")
        return None, None

    with open(dtype_file, "r") as f:
        data_types = json.load(f)

    dtype = {col: typ for col, typ in data_types.items() if typ not in ['datetime64[ns]', 'timedelta64[ns]']}
    parse_dates = [col for col, typ in data_types.items() if typ == 'datetime64[ns]']

    # Load athlete data
    data_file = os.path.join(base_dir, '../data/ams_marathon2024_processed.csv')
    if not os.path.isfile(data_file):
        st.error(f"Data file not found at {data_file}")
        return None, None

    athlete_data = pd.read_csv(data_file, dtype=dtype, parse_dates=parse_dates)

    # Convert columns with timedelta64[ns] type
    timedelta_columns = [col for col, typ in data_types.items() if typ == 'timedelta64[ns]']
    for col in timedelta_columns:
        athlete_data[col] = pd.to_timedelta(athlete_data[col])

    return athlete_data, data_types
