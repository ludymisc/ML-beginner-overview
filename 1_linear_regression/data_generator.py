import numpy as np
import pandas as pd

def generator_train (num_samples=100):
    np.random.seed(11)
    kualitas_level = np.random.randint(1, 10, num_samples)
    brand_level = np.random.randint(1, 5, num_samples)

    samples = pd.DataFrame({
        'kualitas_level': kualitas_level,
        'brand_level': brand_level,
        'harga': 1000 + (kualitas_level * 189) + (brand_level * 322)
    })

    return samples.to_csv('./ML_beginner_material/1_linear_regression/train.csv', index=False)

def generator_test (num_samples=100):
    np.random.seed(42)
    kualitas_level = np.random.randint(1, 10, num_samples)
    brand_level = np.random.randint(1, 5, num_samples)

    samples = pd.DataFrame({
        'kualitas_level': kualitas_level,
        'brand_level': brand_level,
        'harga': 1000 + (kualitas_level * 189) + (brand_level * 322)
    })

    full_df = samples.to_csv('./ML_beginner_material/1_linear_regression/test.csv', index=False)
    test_df = samples.drop('harga', axis=1).to_csv('./ML_beginner_material/1_linear_regression/test_true.csv', index=False)

    return full_df, test_df

if __name__ == "__main__":
    generator_test()
    generator_train()
    print("data generated succesfully")


generator_train(5000)
generator_test(5000)