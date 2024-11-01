import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigcode/tiny_starcoder_py"
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_completion(prefix, suffix, max_new_tokens=50):
    input_text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
    completion = tokenizer.decode(outputs[0])
    return completion

# Load dataset and generate completions
with open("code_completion_dataset.json", "r") as f:
    dataset = json.load(f)

results = []
for example in dataset:
    completion = generate_completion(example["prefix"], example["suffix"])
    # From <fim_middle> to end
    fill = completion[completion.index("<fim_middle>") + len("<fim_middle>"):]
    # Extract first non empty line
    lines = fill.splitlines()
    # Find the first non-empty line and isnt comment
    first_non_empty_line = next((line for line in lines if line.strip() and not line.strip().startswith("#")), None)

    fill = first_non_empty_line

    # Remove "<|endoftext|>" if present
    fill = fill.replace("<|endoftext|>", "")

    generated_code = f'{example["prefix"]}{fill}{example["suffix"]}'
    original_code = f'{example["prefix"]}{example["middle"]}{example["suffix"]}'
    results.append({
        #"prefix": example["prefix"],
        "middle": example["middle"],
        #"suffix": example["suffix"],
        "fill": fill,
        "original_code": original_code,
        "generated_code": generated_code,
        "correct": ""
    })

with open("code_completion_results.json", "w") as f:
    json.dump(results, f, indent=4)
