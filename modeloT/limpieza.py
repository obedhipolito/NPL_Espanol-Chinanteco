import random

def load_and_prepare_data(text_file):
    with open(text_file) as f:
        lines = f.read().split("\n")[:-1]
    text_pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 2:
            chi, spa = parts
            chi = chi.lower()
            spa = spa.lower()
            text_pairs.append((chi, spa))
        else:
            print(f"Warning: Skipping malformed line: {line}")

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    return train_pairs, val_pairs, test_pairs

def print_data_stats(train_pairs, val_pairs, test_pairs):
    print(f"{len(train_pairs) + len(val_pairs) + len(test_pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")

def main():
    text_file = "./spa.txt"
    train_pairs, val_pairs, test_pairs = load_and_prepare_data(text_file)
    print_data_stats(train_pairs, val_pairs, test_pairs)
    return train_pairs, val_pairs, test_pairs

if __name__ == "__main__":
    main()