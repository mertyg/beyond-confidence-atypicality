import torch
import transformers
import torch.nn.functional as F

class HFModel:
    def __init__(self,model, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
    
    @torch.no_grad()
    def get_batch_loglikelihood(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        labels = tokenized.input_ids
        outputs = self.model(**tokenized, labels=labels)
        logits = outputs.logits.cpu()
        labels = labels.cpu()
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none").detach()
        ll_per_sample = -loss.view(shift_logits.shape[0], shift_logits.shape[1])
        nonpad_per_row = (shift_labels != -100).sum(dim=1)
        ll_mean = ll_per_sample.sum(dim=1)/nonpad_per_row
        
        ll_per_sample[(shift_labels == -100)] = 0
        ll_total = ll_per_sample.sum(dim=1)
        torch.cuda.empty_cache()
        return ll_mean.cpu().numpy(), ll_total.cpu().numpy()
    
    @torch.no_grad()
    def generate(self, texts, *args, **kwargs):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        generated = self.model.generate(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"], *args, **kwargs)
        torch.cuda.empty_cache()
        return generated

def get_alpaca7b(model_path, **kwargs):
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = model.to("cuda")
    model = model.eval()
    model_wrapped = HFModel(model, tokenizer)
    model_wrapped.model_name = "Alpaca7b"
    return model_wrapped

def get_llm(model_name, **kwargs):
    if model_name.lower() == "alpaca7b":
        return get_alpaca7b(**kwargs)