import pandas as pd
csv_file="./pet_adoption_data.csv"
output_path="./jsondataset.jsonl"

df=pd.read_csv(csv_file)
try:    
    df.to_json(output_path,orient="records",lines=True)
    print("Conversion from csv to jsonl success!")
except :
    print("failed")