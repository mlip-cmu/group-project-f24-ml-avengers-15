import pandas as pd
import numpy as np
import joblib
from surprise import Dataset, Reader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(group):
    return np.sqrt(mean_squared_error(group['true_rating'], group['predicted_rating']))

def calculate_mae(group):
    return mean_absolute_error(group['true_rating'], group['predicted_rating'])

def save_file(rmse, mae):
    with open("evaluation_metrics.txt", "w") as f:
        f.write(f"Offline Evaluation RMSE: {str(rmse)}\n")
        f.write(f"Offline Evaluation MAE: {str(mae)}")

if __name__ == "__main__":
    data = pd.read_csv('../data/combined_dataset.csv')
    
    # Time-based split
    train_size = int(len(data) * 0.8)
    train_df, test_df = data[:train_size], data[train_size:]

    # Initialize Reader object
    reader = Reader(rating_scale=(1, 5))

    # Create Surprise Dataset objects
    train_data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
    test_data = Dataset.load_from_df(test_df[["user_id", "movie_id", "rating"]], reader)

    trainset = train_data.build_full_trainset()
    testset = test_data.build_full_trainset().build_testset()

    model = joblib.load('../models/SVD_movie_recommender.pkl')
    # predictions = model.test(trainset)
    predictions = model.test(testset)

    # Data Leakage Prevention
    train_df_surprise = pd.DataFrame(train_data.raw_ratings, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    test_df_surprise = pd.DataFrame(test_data.raw_ratings, columns=['user_id', 'movie_id', 'rating', 'timestamp'])

    # Load the original dataset with all demographic information
    original_data = pd.read_csv('../data/combined_dataset.csv')

    # Merge demographic information with test data
    test_df_complete = test_df_surprise.merge(original_data[['user_id', 'age', 'occupation', 'gender']], 
                                            on='user_id', 
                                            how='left')
    if test_df_complete.isnull().values.any():
        print('Warning: There are missing demographic entries in the test set.')

    # Check for data leakage
    train_ids = set(train_df_surprise['user_id'])
    test_ids = set(test_df_surprise['user_id'])
    data_leakage = train_ids.intersection(test_ids)
    if data_leakage:
        print(f'Warning: Data leakage detected with {len(data_leakage)} overlapping user IDs!')
        test_df_surprise = test_df_surprise[~test_df_surprise['user_id'].isin(data_leakage)]
    else:
        print('No data leakage detected.')

    testset = [(uid, iid, r) for (uid, iid, r, _) in test_df_surprise.itertuples(index=False)]
    predictions = model.test(testset)

    # We need to create a DataFrame with user_id, movie_id, true_rating, and predicted_rating
    predictions_df = pd.DataFrame({
        'user_id': [pred.uid for pred in predictions],
        'movie_id': [pred.iid for pred in predictions],
        'true_rating': [pred.r_ui for pred in predictions],
        'predicted_rating': [pred.est for pred in predictions]
    })

    # Merge predictions with test_df_complete to get demographic information
    merged_df = pd.merge(predictions_df, test_df_complete, on=['user_id', 'movie_id'])

    # 1. Analysis by Occupation
    print("Performance by Occupation:")
    occupation_performance_rmse = merged_df.groupby('occupation').apply(calculate_rmse).sort_values()
    occupation_performance_mae = merged_df.groupby('occupation').apply(calculate_mae).sort_values()
    for occupation, rmse in occupation_performance_rmse.items():
        occupation_count = merged_df[merged_df['occupation'] == occupation].shape[0]
        mae = occupation_performance_mae[occupation]
        print(f"Occupation: {occupation}")
        print(f"Number of predictions: {occupation_count}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print("---")

    print("\n" + "="*50 + "\n")

    # 2. Analysis by Gender
    print("Performance by Gender:")
    gender_performance_rmse = merged_df.groupby('gender').apply(calculate_rmse).sort_values()
    gender_performance_mae = merged_df.groupby('gender').apply(calculate_mae).sort_values()
    for gender, rmse in gender_performance_rmse.items():
        gender_count = merged_df[merged_df['gender'] == gender].shape[0]
        mae = gender_performance_mae[gender]
        print(f"Gender: {gender}")
        print(f"Number of predictions: {gender_count}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print("---")

    print("\n" + "="*50 + "\n")

    # 3. Analysis by Age Groups
    merged_df['age_group'] = pd.cut(merged_df['age'], 
                                    bins=[0, 18, 25, 35, 50, 100],
                                    labels=['Under 18', '18-25', '26-35', '36-50', 'Over 50'])

    print("Performance by Age Group:")
    age_group_performance_rmse = merged_df.groupby('age_group').apply(calculate_rmse).sort_values()
    age_group_performance_mae = merged_df.groupby('age_group').apply(calculate_mae).sort_values()
    for age_group, rmse in age_group_performance_rmse.items():
        age_group_count = merged_df[merged_df['age_group'] == age_group].shape[0]
        mae = age_group_performance_mae[age_group]
        print(f"Age Group: {age_group}")
        print(f"Number of predictions: {age_group_count}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print("---")

    # 4. Overall Performance
    overall_rmse = calculate_rmse(merged_df)
    overall_mae = calculate_mae(merged_df)
    save_file(overall_rmse, overall_mae)
    print(f"\nOverall RMSE: {overall_rmse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")

    # Justification of MAE
    print("\nJustification of MAE:")
    print("MAE is suitable for time-series data as it is robust to outliers and gives a linear score, making it easier to interpret.")
    print("It avoids common pitfalls of random splits by ensuring that the model is evaluated on sequential data, preserving the temporal order.")