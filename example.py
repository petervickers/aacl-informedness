import numpy as np
from metrics.ema_nit import normalised_information_transfer, entropy_modulated_accuracy
from metrics.informedness import informedness
from sklearn.metrics import accuracy_score
import argparse

def simulate_and_evaluate(trials, p, model_ability):
    coin_flips = np.random.binomial(n=1, p=p, size=trials)
    prior_guess = np.random.binomial(n=1, p=p, size=trials)


    prior_choices = np.random.binomial(n=1, p=p, size=trials)
    model_preds = [coin_flips[idx] if np.random.rand((1))<model_ability else \
                      prior_choices[idx] for idx in range(len(coin_flips))]

         

    informedness_score = informedness(coin_flips, model_preds)
    nit_score = normalised_information_transfer(coin_flips, model_preds)
    ema_score = entropy_modulated_accuracy(coin_flips, model_preds)
    accuracy = accuracy_score(coin_flips, model_preds)

    return informedness_score, nit_score, ema_score, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metrics on simulated data")
    parser.add_argument("--trials", type=int, default=1000, help="Number of two-class events to simulate")
    parser.add_argument("--binary_prior", type=float, default=0.5, help="Probability of X=1")
    parser.add_argument("--model_ability", type=float, default=0.5, help="Model's ability")

    args = parser.parse_args()

    informedness_score, nit_score, ema_score, accuracy = simulate_and_evaluate(args.trials, args.binary_prior, args.model_ability)

    print(f"Informedness: {informedness_score:.4f}")
    print(f"Normalized Information Transfer: {nit_score:.4f}")
    print(f"Entropy Modulated Accuracy: {ema_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
