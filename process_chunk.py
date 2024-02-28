import csv
import os
import subprocess
import json
import os
import shutil
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def process_prompt(prompt):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Tokenize the prompt
    tokens = word_tokenize(prompt)

    # Remove stop words and lemmatize with POS tagging
    processed_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            pos = get_wordnet_pos(token)
            processed_token = lemmatizer.lemmatize(token, pos)
            processed_tokens.append(processed_token)

    return ' '.join(processed_tokens)

def load_and_process_prompts(prompt_file):
    with open(prompt_file, 'r') as file:
        prompts = json.load(file)
    
    return {prompt['original_class']: process_prompt(prompt['prompt']) for prompt in prompts}

def purge_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


# Initialize a counter for each event to create unique filenames
prompts = load_and_process_prompts('exp3_classes.json')
classes = list(prompts.keys())
print(classes)
event_counters = {event: 0 for event in classes}
output_dir = "/home/storage/chunked_db_10percent"
metadata_dir = "metadata"
data_jsonl_dir = "./egs/cochldb"
os.makedirs(output_dir, exist_ok=True)
purge_directory(metadata_dir)
purge_directory(data_jsonl_dir)

def extract_audio_segment(input_path, start_time, end_time, output_path, event_name, split):
    """
    Uses ffmpeg to extract a segment from an audio file without loading the entire file into memory.
    """
    duration = round(float(end_time) - float(start_time), 2)
    if duration < 0.5 or duration > 60:
        return

    # Construct the ffmpeg command to extract a segment
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i", input_path,  # Input file
        "-ss", str(start_time),  # Start time
        "-to", str(end_time),  # End time
        "-ar", "32000",  # Convert audio sample rate to 32kHz
        output_path  # Output file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Append data to data.jsonl
    data = {
        "path": os.path.abspath(output_path),
        "duration": duration,
        "sample_rate": 32000,
        "amplitude": None,
        "weight": None,
        "info_path": None
    }
    
    # Writing to the appropriate split's data.jsonl
    split_path = os.path.join(data_jsonl_dir, split, 'data.jsonl')
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')
    
    # Generate metadata
    metadata_file_path = os.path.join(metadata_dir, f"{os.path.basename(output_path).replace('.wav', '.json')}")
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump({"description": prompts[event_name]}, f, ensure_ascii=False)

# Process the TSV file
with open("data_table.tsv", "r") as tsv:    
    for line in tsv:
        parts = line.strip().split('\t')
        if len(parts) in [6, 7]:
            file_path, event_name, start, end, _, split = parts[:6]
        else:
            continue

        if event_name not in classes:
            continue
        event_counters[event_name] += 1
        event_folder = os.path.join(output_dir, event_name)
        os.makedirs(event_folder, exist_ok=True)
        new_filename = f"{event_name}_{event_counters[event_name]}.wav"
        print(new_filename)
        new_file_path = os.path.join(event_folder, new_filename)
        
        # Extract and process audio segment
        extract_audio_segment(file_path, start, end, new_file_path, event_name, split)
                    
print("Processing complete.")
