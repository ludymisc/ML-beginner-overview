import numpy as np
import pandas as pd

def sigmoid(x):
    result =  1 / (1 + np.exp(-x))
    return result

def generate_churn_data(n=100, seed=42):
    np.random.seed(seed)

    tenure = np.random.randint(1, 37, n)
    charges = np.random.randint(100, 501, n)

    score = (
        -0.15 * tenure +
        0.01 * charges -
        2
    )

    prob_churn = sigmoid(score)
    churn = np.random.binomial(1, prob_churn)

    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': charges,
        'churn': churn
    })

    csvied_df = df.to_csv('./ML_beginner_material/2_logistic_regression/train.csv', index=False)

    return csvied_df

def generate_churn_data_test(n=100, seed=12):
    np.random.seed(seed)

    tenure = np.random.randint(1, 37, n)
    charges = np.random.randint(100, 501, n)

    score = (
        -0.15 * tenure +
        0.01 * charges -
        2
    )

    prob_churn = sigmoid(score)
    churn = np.random.binomial(1, prob_churn)

    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': charges,
        'churn': churn
    })

    csvied_df_true = df.to_csv('./ML_beginner_material/2_logistic_regression/test_true.csv', index=False)
    csvied_df = df.drop('churn', axis=1).to_csv('./ML_beginner_material/2_logistic_regression/test.csv', index=False)

    return csvied_df, csvied_df_true

if __name__ == "__main__":
    generate_churn_data()
    generate_churn_data_test()
    print("success")


generate_churn_data(5000)
generate_churn_data_test(5000)