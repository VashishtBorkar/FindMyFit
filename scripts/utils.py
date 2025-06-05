from sklearn.model_selection import train_test_split

def split_pairs(pairs, train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    train_pairs, temp_pairs = train_test_split(pairs, train_size=train_size, random_state=seed)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=test_size / (val_size + test_size), random_state=seed)
    
    return train_pairs, val_pairs, test_pairs