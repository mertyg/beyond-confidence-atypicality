import numpy as np
from datasets import load_dataset
from tqdm.contrib.concurrent import thread_map

def load_imdb(filter_tokenizer, filter_by=500, **kwargs):
    test_dataset = load_dataset("imdb", split="test")
    test_sentences = [item["text"] for item in test_dataset]
    test_labels = [item["label"] for item in test_dataset]

    if filter_tokenizer is not None:
        print(f"Filtering by {filter_by} tokens.")
        fn = lambda t: len(filter_tokenizer.encode(t))
        token_lengths_test = np.array(thread_map(fn, test_sentences, max_workers=8))
        test_sentences = list(np.array(test_sentences)[token_lengths_test <= filter_by])
        test_labels = list(np.array(test_labels)[token_lengths_test <= filter_by])

    params = {}
    params['prompt_prefix'] = ""
    params["q_prefix"] = "The following review was written for a movie: "
    params["a_prefix"] = "What is the sentiment, Positive or Negative? Answer: "
    params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
    params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
    params['num_tokens_to_predict'] = 1
    return None, None, test_sentences, np.array(test_labels), params


def load_agnews(**kwargs):
    dataset = load_dataset("ag_news")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_sentences = [item["text"] for item in train_dataset]
    train_labels = [item["label"] for item in train_dataset]
    test_sentences = [item["text"] for item in test_dataset]
    test_labels = [item["label"] for item in test_dataset]
    params = {}
    params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
    params["q_prefix"] = "Article: "
    params["a_prefix"] = "Answer: "
    params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Science and Technology']}
    params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Science and Technology': 3} # notice index start from 1 here
    params['num_tokens_to_predict'] = 1
    return train_sentences, np.array(train_labels), test_sentences, np.array(test_labels), params


def load_trec(**kwargs):
    dataset = load_dataset("trec")
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_sentences = [item["text"] for item in train_dataset]
    train_labels = [item["coarse_label"] for item in train_dataset]
    test_sentences = [item["text"] for item in test_dataset]
    test_labels = [item["coarse_label"] for item in test_dataset]
    params = {}
    params['prompt_prefix'] = "Classify the questions based on their Answer Type. Potential Answer Types are: Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
    params["q_prefix"] = "Question: "
    params["a_prefix"] = "Answer Type: "
    params['label_dict'] = {0: ['Abbreviation'], 1: ['Entity'], 2: ['Description'], 3: ['Human'], 4: ['Location'], 5: ['Number']}
    params['inv_label_dict'] = {'Abbreviation': 0, 'Entity': 1, 'Description': 2, 'Human': 3, 'Location': 4, 'Number': 5}
    params['num_tokens_to_predict'] = 1
    return train_sentences, np.array(train_labels), test_sentences, np.array(test_labels), params
    

def get_llm_classification_data(dataset_name, **kwargs):
    if dataset_name == "imdb":
        return load_imdb(**kwargs)
    elif dataset_name == "agnews":
        return load_agnews(**kwargs)
    elif dataset_name == "trec":
        return load_trec(**kwargs)
    else:
        raise ValueError("Dataset is not available")