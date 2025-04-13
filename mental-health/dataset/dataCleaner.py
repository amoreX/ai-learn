import json

# Load the original JSON file
with open("all_data.json", "r") as f:
    data = json.load(f)

# Remove empty key-value pairs from each dictionary
cleaned_data = []
for item in data:
    cleaned_item = {k: v for k, v in item.items() if k.strip() != ""}
    cleaned_data.append(cleaned_item)

# Save the cleaned JSON
with open("all_data.json", "w") as f:
    json.dump(cleaned_data, f, indent=4)
