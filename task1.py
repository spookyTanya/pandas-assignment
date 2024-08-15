import pandas as pd
from os.path import exists

pd.set_option('display.max.columns', 20)


def print_dataframe_info(data: pd.DataFrame, message=None):
    if message:
        print(message)

    print("Shape:", data.shape)
    print("\nNumber of missing values per column")
    print(data.isna().sum())
    print("\nData statistics:")
    print(data.describe())
    print("\nFirst 10 rows:", data.head(10))


def load_data() -> pd.DataFrame:
    return pd.read_csv('AB_NYC_2019.csv')


def fill_na(data: pd.DataFrame) -> pd.DataFrame:
    data[['name', 'host_name']] = data[['name', 'host_name']].fillna(value='Unknown')
    data['last_review'] = data['last_review'].fillna(value='NaT')
    return data


def add_columns(data: pd.DataFrame):
    data['price_category'] = data['price'].apply(lambda x: 'Low' if x < 100 else ('Medium' if x < 300 else 'High'))
    data['length_of_stay_category'] = data['minimum_nights'].apply(
        lambda x: 'Short-term' if x < 4 else ('Medium-term' if x < 14 else 'Long-term'))


def remove_zero_prices(data: pd.DataFrame):
    data.drop(data[data['price'] == 0].index, inplace=True)


df = load_data()

print_dataframe_info(df, "Initial data:")

print("Info:")
df.info()


# 2
empty_host_index = df[df['host_name'].isna()].index[0]
df = fill_na(df)
print_dataframe_info(df)
assert len(df[df['last_review'].isna()]) == 0, "Shouldn't have missing values in last_review column"
assert df.iloc[empty_host_index]['host_name'] == 'Unknown', "Should have filled host_name"


# 3
add_columns(df)
print_dataframe_info(df)
unique_categories = df['length_of_stay_category'].unique()
assert all(x in ['Short-term', 'Medium-term', 'Long-term'] for x in unique_categories), 'Values should be predefined'
assert 'price_category' in df, 'Should add new column'
low_price_index = df[df['price'] == 89].index[0]
assert df.iloc[low_price_index]['price_category'] == 'Low', 'Should set correct category'

# 4
assert len(df[df['last_review'].isna()]) == 0, "Shouldn't have missing values in last_review column"
assert len(df[df['name'].isna()]) == 0, "Shouldn't have missing values in name column"

remove_zero_prices(df)
assert len(df[df['price'] == 0]) == 0, "Should not have rows with price set to 0"
print_dataframe_info(df)

df.to_csv('cleaned_airbnb_data.csv', index=False)
assert exists('cleaned_airbnb_data.csv'), "File must exist after saving"
