import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
from pydub import AudioSegment, silence

import json
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
        
    for class_info in prompts:
        class_info["prompt"] = process_prompt(class_info["prompt"])
        
    return prompts

import librosa
import soundfile as sf
def upsample_class_audios(class_name, nb_audio, input_path, orig_sr):
    print(f"Cleaning {class_name} audios into {input_path} folder")

    for idx in range(0, nb_audio):
        # Load the audio file
        y, sr = librosa.load(f'{input_path}/{class_name}_{idx}.wav', sr=orig_sr)

        y_upsampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

        # overriding same file because of space limits
        sf.write(f"{input_path}/{class_name}_{idx}.wav", y_upsampled, 22050)
        
def cleaning_class_audios(class_name, nb_audio, input_path, output_path, min_length):
    print(f"Cleaning {class_name} audios into {output_path} folder")
    ensure_directory_exists(output_path)

    for idx in range(0, nb_audio):
        audio = AudioSegment.from_wav(f'{input_path}/{class_name}_{idx}.wav')

        chunks = silence.split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_THRESHOLD,
            silence_thresh=SILENCE_THRESHOLD,
            keep_silence=0
        )

        for chunkIdx, chunk in enumerate(chunks):
            if len(chunk) >= min_length:
                chunk.export(f"{output_path}/{class_name}_{idx}_{chunkIdx}.wav", format="wav")



import math
import time
import os

prompts_data = load_and_process_prompts('exp3_classes.json')
MAX_AUDIO_PER_GENERATE = 14

# FIXME: tmp values that won't work on all classes
SILENCE_THRESHOLD = -36
MIN_SILENCE_THRESHOLD = 500
MIN_AUDIO_LENGTH = 500 #ms
GENERATE_DURATION = 10

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


model = AudioGen.get_pretrained('/home/audiogen_project/training/audiocraft/checkpoints/10percent/')
# model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(
    duration=10,
    top_k=100
)  # generate 5 seconds.

output_dir = "./generated_10percent"
print(prompts_data)

generated_counts = {class_info["original_class"]: 0 for class_info in prompts_data}
total_audio_to_generate = sum([class_info["synthetic"] for class_info in prompts_data])
# total_audio_to_generate = sum([1 for class_info in prompts_data])
total_batches = math.ceil(total_audio_to_generate / MAX_AUDIO_PER_GENERATE)
print(f"Total audio to generate: {total_audio_to_generate}")
print(f"Total batches: {total_batches}")

inference_start_time = time.time()

for batch in range(total_batches):
    descriptions = []
    batch_indices = {}  # Keep track of indices within the batch for each class

    for class_info in prompts_data:
        class_name = class_info["original_class"]
        batch_indices[class_name] = generated_counts[class_name]  # Initialize with the current count
        remaining = class_info["synthetic"] - generated_counts[class_info["original_class"]]
        for _ in range(min(remaining, MAX_AUDIO_PER_GENERATE - len(descriptions))):
            descriptions.append(class_info["prompt"])
            generated_counts[class_info["original_class"]] += 1

        if len(descriptions) == MAX_AUDIO_PER_GENERATE:
            break

    print(descriptions)
    wav = model.generate(descriptions)

    for idx, one_wav in enumerate(wav):
        description = descriptions[idx]
        for class_info in prompts_data:
            if class_info["prompt"] == description:
                original_class = class_info["original_class"]
                break

        audio_idx = batch_indices[original_class] 
        batch_indices[original_class] += 1  

        output_path = f'{output_dir}/{original_class}/raw'
        audio_file_path = f"{output_dir}/{original_class}/raw/{descriptions[idx]}_{audio_idx}"
        print(audio_file_path)
        ensure_directory_exists(output_path)

        try:
            audio_write(audio_file_path, one_wav.cpu(), 32000, strategy="loudness", loudness_compressor=True)
        except AssertionError as e:
            print(f"Error while processing {audio_file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error occurred for {audio_file_path}: {e}")
                
inference_end_time = time.time()
print(f"Total inference time: {inference_end_time - inference_start_time:.2f} seconds.")

average_time_per_audio = (inference_end_time - inference_start_time) / total_audio_to_generate
print(f"Average time per audio: {average_time_per_audio:.2f} seconds.")

del model
torch.cuda.empty_cache()

print("Starting to clean generated audios")
for class_info in prompts_data:
    original_class = class_info["original_class"]

    input_path = f'{output_dir}/{original_class}/raw'
    output_path = f'{output_dir}/{original_class}/cleaned'

    # upsample from model's sample rate
    upsample_class_audios(original_class, class_info["synthetic"], input_path, 32000)

    min_length = class_info.get('minimum_length', MIN_AUDIO_LENGTH)
    cleaning_class_audios(original_class, class_info["synthetic"], input_path, output_path, min_length)

