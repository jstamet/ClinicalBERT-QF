"""
Backbone of these helper functions is taken from: https://github.com/OPTML-Group/QF-Attack
"""


import pandas as pd
import numpy as np
import random
import torch
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

def parse_tsv_to_df(file_path: str) -> pd.DataFrame:

    sentences = []
    current_tokens = []
    current_tags = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    token, tag = parts
                    current_tokens.append(token)
                    current_tags.append(tag)
                else:
                    print(f"Skipping malformed line: {line}")
            else:
                if current_tokens:
                    sentences.append({"tokens": current_tokens, "ner_tags": current_tags})
                    current_tokens, current_tags = [], []

    if current_tokens:
        sentences.append({"tokens": current_tokens, "ner_tags": current_tags})

    return pd.DataFrame(sentences)

def tokenize_and_align_labels(examples, tokenizer, MAX_LEN, label2id):
    # `examples["tokens"]` is already split into words
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=MAX_LEN
    )

    labels = []
    for i, words in enumerate(examples["tokens"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        tag_ids = examples["ner_tags"][i]
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, PAD)
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                # Start of a new word
                label_ids.append(label2id[tag_ids[word_idx]])
            else:
                # Subword of a previously seen word
                # Optionally label subwords the same or -100 if you prefer
                label_ids.append(-100)
            prev_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, id2label, seqeval):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens) and convert IDs to labels
    true_labels = [
        [id2label[l] for (l, m) in zip(label_row, pred_row) if l != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]
    pred_labels = [
        [id2label[m] for (l, m) in zip(label_row, pred_row) if l != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]

    results = seqeval.compute(predictions=pred_labels, references=true_labels)
    # 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy'
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def predict_ner_tags(
    text: str,
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    id2label: dict,
    max_length: int = 128
):
    
    model.eval()
    
    device = next(model.parameters()).device
    
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits  

    predictions = torch.argmax(logits, dim=2)  # shape: [1, seq_len]

    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

    pred_label_ids = predictions[0].tolist()

    results = []
    for token, label_id in zip(tokens, pred_label_ids):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
        
        label_str = id2label[label_id]
        results.append((token, label_str))

    return format_ner_tags(results)

def format_ner_tags(tags):
    formatted_tags = []
    for tag in tags:
        if len(tag[0])>2 and (tag[0][0]==tag[0][1]=='#'):
            new_str = tag[0][2:]

            old_item = formatted_tags[-1]
            new_item = (old_item[0] + new_str, old_item[1])
            formatted_tags[-1] = new_item
        else:
            formatted_tags.append(tag)
    return formatted_tags 

seqeval = load_metric("seqeval")

def compute_single_text_metrics(predicted_tags, reference_tags):

    results = seqeval.compute(
        predictions=[predicted_tags],
        references=[reference_tags]
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)



def get_word_embeddings(model, tokenizer, sentence):
    model.eval()
    device = next(model.parameters()).device 
    with torch.no_grad():
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}        
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][0]        
        hidden_states = hidden_states.cpu()
    return hidden_states

def similarity(emb1, emb2):
    vec1 = emb1.mean(dim=0, keepdim=True)  # shape: (1, hidden_dim)
    vec2 = emb2.mean(dim=0, keepdim=True)  # shape: (1, hidden_dim)
    return cos(vec1, vec2).item()


def get_char_table():
    char_table = [
        '·','~','!','@','#','$','%','^','&','*','(',')','=','-','+',
        '.','<','>','?',',','\'',';',':','|','\\','/','_'
    ]
    for i in range(ord('A'), ord('Z')+1):
        char_table.append(chr(i))
    for i in range(ord('a'), ord('z')+1):
        char_table.append(chr(i))
    for i in range(0,10):
        char_table.append(str(i))
    return list(set(char_table))

def add_5_chars(original_text, attack_string):
    return original_text + " " + attack_string

def greedy_5char_attack(original_text, model, tokenizer, char_table):

    orig_emb = get_word_embeddings(model, tokenizer, original_text)
    best_sim = 1.0  # highest possible cos sim is 1.0

    best_attack_chars = [''] * 5
    
    for pos in range(5):
        local_best_char = ''
        local_best_sim = best_sim
        
        for c in char_table:
            best_attack_chars[pos] = c
            
            candidate_attack_string = ''.join(best_attack_chars)
            attacked_text = add_5_chars(original_text, candidate_attack_string)
            
            new_emb = get_word_embeddings(model, tokenizer, attacked_text)
            sim_val = similarity(orig_emb, new_emb)
            
            if sim_val < local_best_sim:
                local_best_sim = sim_val
                local_best_char = c
        

        best_attack_chars[pos] = local_best_char
        best_sim = local_best_sim
    
    final_attack_string = ''.join(best_attack_chars)
    attacked_text = add_5_chars(original_text, final_attack_string)
    
    return attacked_text, best_sim
def init_5char_population(char_table, pop_size=10):
    population = []
    for _ in range(pop_size):
        individual = ''.join(random.choice(char_table) for __ in range(5))
        population.append(individual)
    return population


def fitness_function_5char(model, tokenizer, orig_emb, original_text, candidate_5char):
    attacked_text = add_5_chars(original_text, candidate_5char)
    new_emb = get_word_embeddings(model, tokenizer, attacked_text)
    sim_val = similarity(orig_emb, new_emb)
    return -sim_val  # negative => we want to minimize similarity


def crossover_5char(parent1, parent2):
    idx = random.randint(1, 4)  # 1..4 for a 5-char string
    child1 = parent1[:idx] + parent2[idx:]
    child2 = parent2[:idx] + parent1[idx:]
    return child1, child2


def mutation_5char(child, char_table, mutation_rate=0.1):
    child_list = list(child)
    for i in range(len(child_list)):
        if random.random() < mutation_rate:
            child_list[i] = random.choice(char_table)
    return ''.join(child_list)

def genetic_5char_attack(
    original_text,
    model,
    tokenizer,
    char_table,
    pop_size=10,
    generations=5,
    mutation_rate=0.1
):
    orig_emb = get_word_embeddings(model, tokenizer, original_text)
    
    population = init_5char_population(char_table, pop_size)
    best_individual = population[0]
    best_fitness = fitness_function_5char(model, tokenizer, orig_emb, original_text, best_individual)
    
    for gen in range(generations):
        fitnesses = [
            fitness_function_5char(model, tokenizer, orig_emb, original_text, ind)
            for ind in population
        ]
        
        for i, ind in enumerate(population):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_individual = ind
        
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
        selection_probs = np.exp(-np.arange(pop_size) / (pop_size / 2))
        selection_probs = selection_probs / selection_probs.sum()
        
        new_pop = []
        while len(new_pop) < pop_size:
            idx1, idx2 = np.random.choice(
                sorted_indices, 
                size=2, 
                p=selection_probs,
                replace=False
            )
            p1, p2 = population[idx1], population[idx2]
            
            c1, _ = crossover_5char(p1, p2)
            c1 = mutation_5char(c1, char_table, mutation_rate)
            new_pop.append(c1)
        
        population = new_pop
    
    best_sim = -best_fitness  # since fitness = -similarity
    attacked_text = add_5_chars(original_text, best_individual)
    return attacked_text, best_sim

def genetic_5char_attack2(
    original_text,
    model,
    tokenizer,
    char_table,
    pop_size=10,
    generations=5,
    mutation_rate=0.1
):
    orig_emb = get_word_embeddings(model, tokenizer, original_text)
    
    population = init_5char_population(char_table, pop_size)
    best_individual = population[0]
    best_fitness = fitness_function_5char(model, tokenizer, orig_emb, original_text, best_individual)
    
    for _ in range(generations):
        fitnesses = [
            fitness_function_5char(model, tokenizer, orig_emb, original_text, ind)
            for ind in population
        ]
        
        for i, ind in enumerate(population):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_individual = ind
        
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        top_k = sorted_pop[: len(sorted_pop)//2]
        
        new_pop = []
        while len(new_pop) < pop_size:
            p1, _ = random.choice(top_k)
            p2, _ = random.choice(top_k)
            c1, c2 = crossover_5char(p1, p2)
            c1 = mutation_5char(c1, char_table, mutation_rate)
            c2 = mutation_5char(c2, char_table, mutation_rate)
            new_pop.append(c1)
            new_pop.append(c2)
        population = new_pop[:pop_size]
    
    best_attack_string = best_individual
    best_sim = -best_fitness  # since fitness = -similarity
    attacked_text = add_5_chars(original_text, best_attack_string)
    return attacked_text, best_sim

class PGD5CharAttacker:
    def __init__(self, model, tokenizer, char_table, max_iter=20):
        self.model = model
        self.tokenizer = tokenizer
        self.char_table = char_table
        self.max_iter = max_iter
    
    def attack(self, original_text):
        orig_emb = get_word_embeddings(self.model, self.tokenizer, original_text)
        
        best_chars = [''] * 5
        best_sim = 1.0
        
        for _ in range(self.max_iter):
            improved = False
            for pos in range(5):
                current_char = best_chars[pos]
                local_best_char = current_char
                local_best_sim = best_sim
                
                for c in self.char_table:
                    # Temporarily put c at pos
                    best_chars[pos] = c
                    # Check new similarity
                    candidate_string = ''.join(best_chars)
                    attacked_text = add_5_chars(original_text, candidate_string)
                    new_emb = get_word_embeddings(self.model, self.tokenizer, attacked_text)
                    sim_val = similarity(orig_emb, new_emb)
                    
                    if sim_val < local_best_sim:
                        local_best_sim = sim_val
                        local_best_char = c
                
                best_chars[pos] = local_best_char
                if local_best_sim < best_sim:
                    best_sim = local_best_sim
                    improved = True
            
            if not improved:
                break
        
        best_string = ''.join(best_chars)
        attacked_text = add_5_chars(original_text, best_string)
        return attacked_text, best_sim
def evaluate_attacks(
    original_text: str,
    attack_texts: dict,
    model,
    tokenizer,
    id2label: dict,
):

    original_preds = predict_ner_tags(
        text=original_text,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label
    )
    orig_pred_tokens, orig_pred_tags = zip(*original_preds) if original_preds else ([], [])


    results = {}
    
    results["original"] = {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "accuracy": 1.0,
    }
    
    for attack_name, attacked_str in attack_texts.items():
        attacked_preds = predict_ner_tags(
            text=attacked_str,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label
        )
        
        att_pred_tokens, att_pred_tags = zip(*attacked_preds) if attacked_preds else ([], [])
        

        len_orig = len(orig_pred_tags)
        if len(att_pred_tags) < len_orig:
            print(f"Warning: attacked prediction has fewer tokens ({len(att_pred_tags)}) than original ({len_orig}).")
        
        att_pred_tags_short = att_pred_tags[:len_orig]
        orig_pred_tags_short = orig_pred_tags[:len_orig]
        
        # 3. Compute seqeval-based metrics
        attack_metrics = compute_single_text_metrics(
            list(att_pred_tags_short), 
            list(orig_pred_tags_short)
        )
        results[attack_name] = attack_metrics
    
    return results

def compute_average_metrics(total_results, attack_names=None):
    if attack_names is None:
        attack_names = ["original", "greedy", "genetic", "pgd"]
    
    metric_sums = {name: {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0} 
                   for name in attack_names}
    
    count = len(total_results)
    
    for _, eval_results, in total_results:
        for attack_name in attack_names:
            for metric_name in ["precision", "recall", "f1", "accuracy"]:
                metric_sums[attack_name][metric_name] += eval_results[attack_name][metric_name]
                
    for attack_name in attack_names:
        for metric_name in ["precision", "recall", "f1", "accuracy"]:
            metric_sums[attack_name][metric_name] /= count
    
    return metric_sums