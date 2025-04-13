import json

# Input and output file paths
input_file = 'suicide-detection.json'
output_file = 'suicide-detection.jsonl'

# Read the original JSON array
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Write each item as a separate line in the JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item) + '\n')

print("Conversion complete! JSONL saved as", output_file)