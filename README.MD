# Metrics Evaluation

This script allows you to simulate two-class events, generate model predictions based on a specified model ability, and evaluate different metrics including informedness, normalized information transfer, entropy modulated accuracy, and accuracy.

## Usage

1. Make sure you have Python 3 installed.

2. Install the required packages using the following command:

pip install numpy scikit-learn

3. Navigate to the directory containing the `test.py` script.

4. Run the script with the desired parameters using the command line:

python example.py --trials 1000 --binary_prior 0.5 --model_ability 0.7

Replace the values with your desired parameters:
- `--trials`: Number of two-class events to simulate.
- `--binary_prior`: Probability of X=1 (ground truth).
- `--model_ability`: Model's ability to predict.

5. The script will output the calculated metric scores based on the simulated data and predictions.

## Files

- `test.py`: The main script to simulate data and evaluate metrics.
- `metrics/ema_nit.py`: Contains functions for normalized information transfer and entropy modulated accuracy.
- `metrics/informedness.py`: Contains the function for calculating informedness.

## Credits

The metrics functions are inspired by various research works. Refer to the comments in the `ema_nit.py` and `informedness.py` files for references.

For any questions or issues, please feel free to contact [] at [].

