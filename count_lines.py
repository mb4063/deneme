def count_lines(filename):
    count = 0
    with open(filename, 'rb') as f:
        for _ in f:
            count += 1
    return count

if __name__ == "__main__":
    filename = "data/edges_train_A.csv"
    print(f"Counting lines in {filename}...")
    num_lines = count_lines(filename)
    print(f"Total number of lines: {num_lines}") 