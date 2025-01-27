import os
import pandas as pd
import argparse
from autogluon.tabular import TabularDataset, TabularPredictor

# Set up the command line argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Select the chips to be used for training and prediction")
    parser.add_argument(
        '--chips', 
        type=str, 
        nargs='+',  # Allow multiple chips to be passed
        choices=['chip1', 'chip2', 'chip3', 'chip4', 'chip5'],  # Available chip options
        default=['chip1', 'chip2', 'chip3', 'chip4', 'chip5'],  # Default to all chips
        help="Specify which chips to run, for example: chip1 chip3 chip5"
    )
    return parser.parse_args()

def main():
    # Parse the command line arguments
    args = parse_args()

    # User-selected chips
    chips = args.chips
    path = os.path.dirname(os.path.realpath(__file__)) + '/'  # Get the directory of the current script

    for chip in chips:
        # Load the data
        train_data = pd.read_csv(f'{path}{chip}_train_7-3.csv')
        test_data = pd.read_csv(f'{path}{chip}_test_7-3.csv')

        # Calculate the weights
        train_data['weight'] = train_data['label'].apply(lambda x: 5 if x == 1 else 1)

        # Repeat the samples according to the weights
        train_data_weighted = train_data.loc[train_data.index.repeat(train_data['weight'])].reset_index(drop=True)

        # Drop the weight column
        train_data_weighted = train_data_weighted.drop(columns=['weight'])

        # Convert to TabularDataset
        train = TabularDataset(train_data_weighted)
        test = TabularDataset(test_data)

        # Assign a different model path for each chip
        model_path = f'{chip}_predictor'

        # Create and train the predictor
        predictor = TabularPredictor(
            label='label',
            problem_type='binary',
            eval_metric='accuracy',
            path=model_path
        ).fit(
            train_data=train,
            presets='best_quality',
            num_bag_folds=5
        )

        # Evaluate the model
        leaderboard = predictor.leaderboard(test, silent=True)
        print(f'Leaderboard for {chip}:')
        print(leaderboard)

        # Make predictions
        predictions = predictor.predict(test)
        print(f'Predictions for {chip}:')
        print(predictions.head())  # Print the first few prediction results

        print(f'{chip} training and evaluation completed.\n')

if __name__ == "__main__":
    main()
