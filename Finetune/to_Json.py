import pandas as pd
import json
import os

DATA_DIR = os.getenv("DATA_DIR")

def normalize_content_manipulation(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower() 
        if value == 'true': 
            return True
        elif value == 'false': 
            return False
    return None

    import pandas as pd

def normalize_content_manipulation(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value == 'true':
            return True
        elif value == 'false':
            return False
    return None

train_df = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset/train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset/test.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'final_dataset/val.csv'))
train_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

def duplicate_true_rows(df):
    df['content_manipulation'] = df['content_manipulation'].apply(normalize_content_manipulation)
    true_rows = df[df['content_manipulation'] == True]
    duplicated = pd.concat([df, true_rows, true_rows], ignore_index=True)
    return duplicated

train_df = duplicate_true_rows(train_df)
train_df = train_df.sample(frac=1).reset_index(drop=True)



# Initialize an empty list to store the conversation data
conversations = []

# Iterate through each row in the DataFrame
for i in range(len(train_df)):
    content_manipulation = train_df.iloc[i]['content_manipulation']
    
    # Skip rows where 'content_manipulation' is None
    if content_manipulation is None:
        continue
    
    # Construct the user query and assistant response
    query = "Is this image manipulated or synthesized?"
    if content_manipulation:
        response = f"This image has been manipulated or synthesized. {train_df.iloc[i]['prompts']}"
    else:
        response = f"This image has not been manipulated or synthesized. {train_df.iloc[i]['prompts']}"

    image_path = f"{DATA_DIR}/final_dataset/images/{train_df.iloc[i]['image_id']}.jpg"
    
    # Create the conversation entry
    conversation = {
        "id": f"identity_{i+1}",
        "conversations": [
            {
                "from": "user",
                "value": f"{query} <|vision_start|>{image_path}<|vision_end|>"
            },
            {
                "from": "assistant",
                "value": response
            }
        ]
    }
    
    # Append the conversation to the list
    conversations.append(conversation)

# Convert the result list to a JSON string
json_output = json.dumps(conversations, ensure_ascii=False, indent=4)

# Output the result to a JSON file
output_path = "./Ray_Job/train.json"
with open(output_path, 'w') as f:
    f.write(json_output)