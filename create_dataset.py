import os
import json
import random

def load_files_from_directory(directory, extension=".py"):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            with open(os.path.join(directory, filename), 'r') as file:
                files.append(file.read())
    return files

def generate_code_completions_dataset(files, num_examples=30):
    examples = []
    blacklist = ["#", "print("]
    cursor_pos_list = []
    for code in files:
        lines = code.splitlines()
        for _ in range(num_examples // len(files)):
            if len(lines) < 3: 
                continue
            while True:
                cursor_pos = random.randint(1, len(lines) - 2)
                if cursor_pos in cursor_pos_list:
                    continue
                if lines[cursor_pos].strip() == "":
                    continue
                # If the line starts with //
                if lines[cursor_pos].strip().startswith("//"):
                    continue
                # If the line is too short
                if len(lines[cursor_pos].strip()) < 5:
                    continue

                if not any((b in lines[cursor_pos].strip()) for b in blacklist):
                    break
            cursor_pos_list.append(cursor_pos)
            prefix = "\n".join(lines[:cursor_pos]) + "\n"
            middle = lines[cursor_pos]
            suffix = "\n".join(lines[cursor_pos + 1:])
            examples.append({"prefix": prefix, "middle": middle, "suffix": suffix})
    return examples

files = load_files_from_directory('examples')
dataset = generate_code_completions_dataset(files)

with open("code_completion_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)