import pandas as pd
from os.path import exists

pd.set_option('display.max.columns', 20)


def display_grouped_data(data: pd.DataFrame, message=None):
    if message:
        print(message)

    print("First 10 rows:\n", data.head(10))


def load_data() -> pd.DataFrame:
    return pd.read_csv('cleaned_airbnb_data.csv')


def select_data(data: pd.DataFrame):
    print("Selected row using .loc\t", data.loc[0])
    print("Selected row using .loc\n", data.loc[3:6])
    assert data.loc[0].equals(data.iloc[0]), 'The results from .loc and .iloc should match'
    print("Selected column values using .loc:\n", data.loc[1, ['name', 'price']])
    print("Selected single column value using .iloc:\t", data.iloc[2, 1])
    print("Selected column values using .iloc:\n", data.iloc[2, [1, 9]])
    assert data.loc[1, 'name'] == data.iloc[1, 1], "The values from .loc and .iloc should match"


def filter_by_neighbourhood_group(data: pd.DataFrame, neighbourhood_group: str) -> pd.DataFrame:
    return data[data['neighbourhood_group'] == neighbourhood_group]


def filter_by_price_and_reviews(data: pd.DataFrame) -> pd.DataFrame:
    return data.query('price > 100 and number_of_reviews > 10')


def add_ranks(data: pd.DataFrame) -> pd.DataFrame:
    copy = data.copy()
    grouped = copy.groupby('neighbourhood_group').agg(
        total_items=('id', 'count'),
        mean_price=('price', 'mean')
    )
    grouped['rank_listings'] = grouped['total_items'].rank(ascending=False)
    grouped['rank_price'] = grouped['mean_price'].rank(ascending=False)
    grouped['combined_rank'] = (grouped['rank_listings'] + grouped['rank_price']).rank()
    grouped.sort_values('combined_rank', inplace=True)
    return grouped


df = load_data()
display_grouped_data(df, 'Initial Data:')

# 1
select_data(df)
filtered = filter_by_neighbourhood_group(df, "Brooklyn")
display_grouped_data(filtered, "Filtered by neighbourhood:")
assert len(filtered[filtered['neighbourhood_group'] == "Manhattan"]) == 0, "Should skip listings from other neighbourhoods"

filtered = filter_by_price_and_reviews(df)
display_grouped_data(filtered, "Filtered by price and reviews:")
assert len(filtered[filtered['price'] < 100]) == 0, "Should not include listings where price is lower than 100"

columns = filtered.loc[:, ['neighbourhood_group', 'price', 'minimum_nights', 'number_of_reviews', 'price_category', 'availability_365']]
display_grouped_data(columns, "Selected columns for further analysis:")
assert columns.shape[1] == 6, "Should have 6 columns"


# 2
display_grouped_data(columns.groupby(['neighbourhood_group', 'price_category']).mean())
print("Average values", columns.groupby(['neighbourhood_group', 'price_category']).mean())


# 3
sorted_data = columns.sort_values(['price', 'number_of_reviews'], ascending=[0, 1])
display_grouped_data(sorted_data, "Sorted data")
assert sorted_data.iloc[0]['price'] >= sorted_data.iloc[1]['price'], "Price should be sorted in descending order"

ranked = add_ranks(filtered)
display_grouped_data(ranked, "Ranked data based on the total number of listings and the average price")
assert 'rank_listings' in ranked.iloc[0], "Should have listing rank"
assert 'combined_rank' in ranked.iloc[0], "Should have combined rank"
assert ranked.iloc[0]['combined_rank'] <= ranked.iloc[1]['combined_rank'], "Should be sorted by combined rank"

ranked.to_csv('aggregated_airbnb_data.csv', index=False)
assert exists('aggregated_airbnb_data.csv'), "File must exist after saving"
