import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax

def construct_prompt(params, test_sentence):
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    
    return prompt

def chunks(lst, bs):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]

class RecordCollection:
    def __init__(self, keys):
        self.keys = keys
        self.records = {}
        for k in self.keys:
            self.records[k] = []
    
    def update(self, record):
        for k in self.keys:
            self.records[k].append(record[k])
    
    def collect(self):
        all_records = {}
        for k in self.keys:
            all_records[k] = np.concatenate(self.records[k])
        return all_records


@torch.no_grad()
def run_texts(model, texts, label_tokens):
    """
    Runs a list of texts through the model and returns the logits for the label tokens.
    """
    completions = model.generate(texts, max_new_tokens=1, do_sample=False, pad_token_id=model.tokenizer.eos_token_id,
                                output_scores=True, return_dict_in_generate=True)
    
    token_logits = completions.scores[0].cpu().numpy()
    label_token_logits = token_logits[:, label_tokens]
    normalized_label_token_probs = softmax(label_token_logits, axis=1)
    overall_label_token_probs = softmax(token_logits, axis=1)[:, label_tokens]
    
    model_preds = np.argmax(label_token_logits, axis=1)
    _, prompt_total_logprob = model.get_batch_loglikelihood(texts)
    
    records = {"normalized_probs": np.max(normalized_label_token_probs, axis=1),
               "overall_probs": np.max(overall_label_token_probs, axis=1),
               "preds": model_preds,
               "atypicality_total_logprob": prompt_total_logprob,
               "logits": label_token_logits}
    return records


def get_experiment_records(model, test_sentences, test_labels, ds_params, batch_size=16):
    prompts = []
    for test_sentence in test_sentences:
        prompts.append(construct_prompt(ds_params, test_sentence))
        
    stats = RecordCollection(["atypicality_total_logprob", "normalized_probs",
                                    "overall_probs", "preds", "logits"])
    
    for batch_prompts in tqdm(chunks(prompts, batch_size)):
        stats.update(run_texts(model, batch_prompts, ds_params["label_tokens"]))

    records = stats.collect()
    records["sample_idx"] = np.array([i for i in range(len(test_sentences))])
    records["labels"] = test_labels
    
    return records

@torch.no_grad()
def get_recalib_predata(model, ds_params):
    label_tokens = ds_params["label_tokens"]
    _, test_label_atypicality = model.get_batch_loglikelihood(list(ds_params["inv_label_dict"].keys()))
    content_free_opts = ['N/A', '', '[MASK]']
    p_cf = []
    for opt in content_free_opts:
        prompts = [construct_prompt(ds_params, opt)]
        completions = model.generate(texts=prompts, max_new_tokens=1, do_sample=False, pad_token_id=model.tokenizer.eos_token_id,
                                    output_scores=True, return_dict_in_generate=True)
        token_logits = completions.scores[0].cpu().numpy()
        label_token_logits = token_logits[:, label_tokens]
        p_cf.append(softmax(label_token_logits, axis=1))
        
    p_cf = np.concatenate(p_cf)
    p_cf = p_cf.mean(axis=0, keepdims=True)
    p_cf = p_cf / np.sum(p_cf) # normalize
    recalib_info = {"test_label_atypicality": test_label_atypicality, "p_cf": p_cf}
    return recalib_info
