import pickle

if __name__ == "__main__":
    # Load data from pkl file
    with open("modelfactory/data/test_data.pkl", "rb") as f:
        data = pickle.load(f)

    # Explore the data
    print(f"Loaded {len(data)} samples")
    print("First sample:", data[0])
    print("Keys in the first sample:", data[0].keys())