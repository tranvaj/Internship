### 1. Creating the input data (create_dataset.py)
To evaluate the quality of "tiny_starcoder" model, first I have to create the input data for the model.

I have taken 3 python files from my own small projects:
- **`dictionary_attack.py`** - simple password cracking attempt using an existing dictionary
- **`knapsack_crypt.py`** - Merkle–Hellman knapsack cryptosystem implementation
- **`tetris.py`** - a small excerpt from my project, which captures game state of a tetris board using OpenCV 

The input data will consist of:
- **prefix** - the line of codes before the line of code that the model will try to predict
- **middle** - the line of code that the model will not see and will try to predict
- **suffix** - the line of code after "middle"

The model will only get inputted with prefix and suffix and it will output a prediction.
(I will evaluate the model on predicting a single line of code)

First I loaded my python files and for each file, I extracted "prefix", "middle" and "suffix", where "middle" is a random line of code that has not been selected yet (to maintain uniqueness of examples). This process is done (30/amount of files) 3 times (for each file) to get 30 examples.

When getting a random line of code (middle), I try to avoid comment lines - comments are subjective, getting an accurate prediction for middle will not give me much information about the quality of the model. I also try to avoid line of codes involving "print()" for the same reasons as stated above. Short lines of codes are skipped to avoid examples such as a line of code containing only braces e.g "}"

All 30 examples are saved in corresponding .json file as a list of dictionaries with keys: "prefix", "middle" and "suffix".

### 2. Creating predictions
Using HF transformers library I can easily initialize the "tiny_starcoder" model.
All the input texts are formatted in this way (with the given special tokens):

´´´"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"´´´

Where the prediction is generated after <fim_middle> token.

When creating a prediction, many times the model generated more tokens than were needed for the prediction (more lines than one). Therefore I extracted the first non-comment line from the prediction. I also removed the "<|endoftext|>" token in the rare case of its appearance. 

All 30 examples went through the prediction and were saved into an appropriate .json format with following keys:
- **middle** - the line of code that the model will try to predict
- **fill** - prediction of the missing line of code
- **original_code** - the original complete code
- **generated_code** - the original complete code with the "middle" replaced with "fill"
- **correct** - is empty string, serves as a key that will describe if the prediction was correct or wrong (manually annotated by me)

### 3. Evaluating performance
To evaluate performance, I used "exact match" (comparing if strings are identical), chrf, BLEU and Levenshtein distance metrics.

When evaluating correlation between my manual annotations of correctness and metrics described above, I found that "exact match" metric correlated the most with my manual annotations. Chrf second most, levenshtein third most (negative correlation because the fewer edits, the higher the correctness) and BLEU the worst. Exact match metric sometimes is not ideal, because the model does give a working, accurate prediction, that is a bit different visually, but functionally the same.

Due to the small amount of examples, I can not say if "exact match" is the best automatic metric for evaluation of this model. 

Intuitivelly, I think that chrf_score or levenshtein distance might be the best metrics if the amount of examples were larger.






