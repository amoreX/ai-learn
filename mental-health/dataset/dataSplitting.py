import json
import random

# File paths
input_file = 'all_data.jsonl'
train_file = 'train.jsonl'
val_file = 'val.jsonl'
test_file = 'test.jsonl'

# Read the JSONL data
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Shuffle the data
random.shuffle(data)

# Split sizes
total = len(data)
train_end = int(0.8 * total)
val_end = train_end + int(0.1 * total)

# Splits
train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# Function to save JSONL
def save_jsonl(filename, dataset):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')

# Save the splits
save_jsonl(train_file, train_data)
save_jsonl(val_file, val_data)
save_jsonl(test_file, test_data)

print(f"Done! {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples saved.")
