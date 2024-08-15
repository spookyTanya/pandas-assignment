import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max.columns', 20)


def print_analysis_results(data: pd.DataFrame, message=None):
    if message:
        print(message)

    print("First 20 rows:\n", data.head(20))


def load_data() -> pd.DataFrame:
    return pd.read_csv('cleaned_airbnb_data.csv')


def create_pivot_table(data: pd.DataFrame) -> pd.DataFrame:
    return pd.pivot_table(data, index='neighbourhood_group', columns='room_type', values='price', aggfunc='mean')


def melt_data(data: pd.DataFrame) -> pd.DataFrame:
    return pd.melt(data, id_vars=['neighbourhood_group'], value_vars=['price', 'minimum_nights'])


def add_availability_column(data: pd.DataFrame):
    data['availability_status'] = data['availability_365'].apply(lambda x: 'Rarely Available' if x < 50 else ('Occasionally Available' if x < 200 else 'Highly Available'))


def get_correlations(data: pd.DataFrame) -> pd.DataFrame:
    copy = data.copy()
    copy['availability_status_code'] = copy['availability_status'].astype('category').cat.codes
    copy['neighbourhood_group_code'] = copy['neighbourhood_group'].astype('category').cat.codes
    correlation_matrix = copy[['availability_status_code', 'price', 'number_of_reviews', 'neighbourhood_group_code', 'minimum_nights']].corr()

    plt.figure(figsize=(15, 8))
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()
    return correlation_matrix


def get_statistics(data: pd.DataFrame):
    price_mean = data['price'].mean()
    price_median = data['price'].median()
    price_std = data['price'].std()
    print(f"Price statistics: mean - {price_mean}, median - {price_median}, std - {price_std}")

    min_nights_mean = data['minimum_nights'].mean()
    min_nights_median = data['minimum_nights'].median()
    min_nights_std = data['minimum_nights'].std()
    print(f"Minimum_nights statistics: mean - {min_nights_mean}, median - {min_nights_median}, std - {min_nights_std}")

    reviews_mean = data['number_of_reviews'].mean()
    reviews_median = data['number_of_reviews'].median()
    reviews_std = data['number_of_reviews'].std()
    print(f"Number_of_reviews statistics: mean - {reviews_mean}, median - {reviews_median}, std - {reviews_std}")

    reviews_per_month_mean = data['reviews_per_month'].mean()
    reviews_per_month_median = data['reviews_per_month'].median()
    reviews_per_month_std = data['reviews_per_month'].std()
    print(f"Reviews_per_month statistics: mean - {reviews_per_month_mean}, median - {reviews_per_month_median}, std - {reviews_per_month_std}")

    availability_mean = data['availability_365'].mean()
    availability_median = data['availability_365'].median()
    availability_std = data['availability_365'].std()
    print(f"Availability statistics: mean - {availability_mean}, median - {availability_median}, std - {availability_std}")


def change_type(data: pd.DataFrame):
    data['last_review_date'] = pd.to_datetime(data['last_review'])


def get_monthly_trends(data: pd.DataFrame) -> pd.DataFrame:
    copy = data.copy()
    copy.set_index('last_review_date', inplace=True)
    return copy.resample('ME').agg({
        'price': 'mean',
        'number_of_reviews': 'sum',
        'minimum_nights': 'mean'
    })


def get_avg_by_month(data: pd.DataFrame) -> pd.DataFrame:
    copy = data.copy()
    copy.set_index('last_review_date', inplace=True)
    return copy.resample('ME').agg({
        'price': 'mean',
        'number_of_reviews': 'mean',
        'minimum_nights': 'mean',
        'reviews_per_month': 'mean',
        'availability_365': 'mean'
    })


df = load_data()

# 1
pivot = create_pivot_table(df)
print_analysis_results(pivot, "Pivoted data")

melted = melt_data(df)
print_analysis_results(melted, "Melted data")

add_availability_column(df)
print_analysis_results(melted, "Data with new availability column")

get_correlations(df)
print_analysis_results(melted, "Correlated data")

# 2
get_statistics(df)

# 3
change_type(df)
print_analysis_results(melted, "Data with converted date field")

monthly_trends = get_monthly_trends(df)
print_analysis_results(monthly_trends, "Monthly trends")

grouped_by_month = get_avg_by_month(df)
print_analysis_results(grouped_by_month, "Monthly averages")
grouped_by_month.to_csv('time_series_airbnb_data.csv', index=False)
