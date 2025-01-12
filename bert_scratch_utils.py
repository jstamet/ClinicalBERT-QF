# """
# Backbone of these helper functions is taken from: https://github.com/OPTML-Group/QF-Attack
# """


# import pandas as pd
# import numpy as np
# import random
# import torch
# from datasets import Dataset, load_metric
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     DataCollatorForTokenClassification,
#     TrainingArguments,
#     Trainer,
# )

# def parse_tsv_to_df(file_path: str) -> pd.DataFrame:

#     sentences = []
#     current_tokens = []
#     current_tags = []

#     with open(file_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 parts = line.split(" ", 1)
#                 if len(parts) == 2:
#                     token, tag = parts
#                     current_tokens.append(token)
#                     current_tags.append(tag)
#                 else:
#                     print(f"Skipping malformed line: {line}")
#             else:
#                 if current_tokens:
#                     sentences.append({"tokens": current_tokens, "ner_tags": current_tags})
#                     current_tokens, current_tags = [], []

#     if current_tokens:
#         sentences.append({"tokens": current_tokens, "ner_tags": current_tags})

#     return pd.DataFrame(sentences)

# def tokenize_and_align_labels(examples, tokenizer, MAX_LEN, label2id):
#     # `examples["tokens"]` is already split into words
#     tokenized_inputs = tokenizer(
#         examples["tokens"],
#         truncation=True,
#         is_split_into_words=True,
#         max_length=MAX_LEN
#     )

#     labels = []
#     for i, words in enumerate(examples["tokens"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         tag_ids = examples["ner_tags"][i]
#         label_ids = []
#         prev_word_idx = None
#         for word_idx in word_ids:
#             if word_idx is None:
#                 # Special tokens (CLS, SEP, PAD)
#                 label_ids.append(-100)
#             elif word_idx != prev_word_idx:
#                 # Start of a new word
#                 label_ids.append(label2id[tag_ids[word_idx]])
#             else:
#                 # Subword of a previously seen word
#                 # Optionally label subwords the same or -100 if you prefer
#                 label_ids.append(-100)
#             prev_word_idx = word_idx

#         labels.append(label_ids)

#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs


# def compute_metrics(p, id2label, seqeval):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)

#     # Remove ignored index (special tokens) and convert IDs to labels
#     true_labels = [
#         [id2label[l] for (l, m) in zip(label_row, pred_row) if l != -100]
#         for label_row, pred_row in zip(labels, predictions)
#     ]
#     pred_labels = [
#         [id2label[m] for (l, m) in zip(label_row, pred_row) if l != -100]
#         for label_row, pred_row in zip(labels, predictions)
#     ]

#     results = seqeval.compute(predictions=pred_labels, references=true_labels)
#     # 'overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy'
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }

# def predict_ner_tags(
#     text: str,
#     model: AutoModelForTokenClassification,
#     tokenizer: AutoTokenizer,
#     id2label: dict,
#     max_length: int = 128
# ):
    
#     model.eval()
    
#     device = next(model.parameters()).device
    
#     encoding = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=max_length,
#     )
    
#     encoding = {k: v.to(device) for k, v in encoding.items()}
    
#     with torch.no_grad():
#         outputs = model(**encoding)
    
#     logits = outputs.logits  

#     predictions = torch.argmax(logits, dim=2)  # shape: [1, seq_len]

#     tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

#     pred_label_ids = predictions[0].tolist()

#     results = []
#     for token, label_id in zip(tokens, pred_label_ids):
#         if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
#             continue
        
#         label_str = id2label[label_id]
#         results.append((token, label_str))

#     return format_ner_tags(results)

# def format_ner_tags(tags):
#     formatted_tags = []
#     for tag in tags:
#         if len(tag[0])>2 and (tag[0][0]==tag[0][1]=='#'):
#             new_str = tag[0][2:]

#             old_item = formatted_tags[-1]
#             new_item = (old_item[0] + new_str, old_item[1])
#             formatted_tags[-1] = new_item
#         else:
#             formatted_tags.append(tag)
#     return formatted_tags 

# seqeval = load_metric("seqeval")

# def compute_single_text_metrics(predicted_tags, reference_tags):

#     results = seqeval.compute(
#         predictions=[predicted_tags],
#         references=[reference_tags]
#     )
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }

# cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)



# def get_word_embeddings(model, tokenizer, sentence):
#     model.eval()
#     device = next(model.parameters()).device 
#     with torch.no_grad():
#         inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=128)
#         inputs = {k: v.to(device) for k, v in inputs.items()}        
#         outputs = model(**inputs, output_hidden_states=True)
#         hidden_states = outputs.hidden_states[-1][0]        
#         hidden_states = hidden_states.cpu()
#     return hidden_states

# def similarity(emb1, emb2):
#     vec1 = emb1.mean(dim=0, keepdim=True)  # shape: (1, hidden_dim)
#     vec2 = emb2.mean(dim=0, keepdim=True)  # shape: (1, hidden_dim)
#     return cos(vec1, vec2).item()


# def get_char_table():
#     char_table = [
#         '·','~','!','@','#','$','%','^','&','*','(',')','=','-','+',
#         '.','<','>','?',',','\'',';',':','|','\\','/','_'
#     ]
#     for i in range(ord('A'), ord('Z')+1):
#         char_table.append(chr(i))
#     for i in range(ord('a'), ord('z')+1):
#         char_table.append(chr(i))
#     for i in range(0,10):
#         char_table.append(str(i))
#     return list(set(char_table))

# def add_5_chars(original_text, attack_string):
#     return original_text + " " + attack_string

# def greedy_5char_attack(original_text, model, tokenizer, char_table):

#     orig_emb = get_word_embeddings(model, tokenizer, original_text)
#     best_sim = 1.0  # highest possible cos sim is 1.0

#     best_attack_chars = [''] * 5
    
#     for pos in range(5):
#         local_best_char = ''
#         local_best_sim = best_sim
        
#         for c in char_table:
#             best_attack_chars[pos] = c
            
#             candidate_attack_string = ''.join(best_attack_chars)
#             attacked_text = add_5_chars(original_text, candidate_attack_string)
            
#             new_emb = get_word_embeddings(model, tokenizer, attacked_text)
#             sim_val = similarity(orig_emb, new_emb)
            
#             if sim_val < local_best_sim:
#                 local_best_sim = sim_val
#                 local_best_char = c
        

#         best_attack_chars[pos] = local_best_char
#         best_sim = local_best_sim
    
#     final_attack_string = ''.join(best_attack_chars)
#     attacked_text = add_5_chars(original_text, final_attack_string)
    
#     return attacked_text, best_sim
# def init_5char_population(char_table, pop_size=10):
#     population = []
#     for _ in range(pop_size):
#         individual = ''.join(random.choice(char_table) for __ in range(5))
#         population.append(individual)
#     return population


# def fitness_function_5char(model, tokenizer, orig_emb, original_text, candidate_5char):
#     attacked_text = add_5_chars(original_text, candidate_5char)
#     new_emb = get_word_embeddings(model, tokenizer, attacked_text)
#     sim_val = similarity(orig_emb, new_emb)
#     return -sim_val  # negative => we want to minimize similarity


# def crossover_5char(parent1, parent2):
#     idx = random.randint(1, 4)  # 1..4 for a 5-char string
#     child1 = parent1[:idx] + parent2[idx:]
#     child2 = parent2[:idx] + parent1[idx:]
#     return child1, child2


# def mutation_5char(child, char_table, mutation_rate=0.1):
#     child_list = list(child)
#     for i in range(len(child_list)):
#         if random.random() < mutation_rate:
#             child_list[i] = random.choice(char_table)
#     return ''.join(child_list)

# def genetic_5char_attack(
#     original_text,
#     model,
#     tokenizer,
#     char_table,
#     pop_size=10,
#     generations=5,
#     mutation_rate=0.1
# ):
#     orig_emb = get_word_embeddings(model, tokenizer, original_text)
    
#     population = init_5char_population(char_table, pop_size)
#     best_individual = population[0]
#     best_fitness = fitness_function_5char(model, tokenizer, orig_emb, original_text, best_individual)
    
#     for gen in range(generations):
#         fitnesses = [
#             fitness_function_5char(model, tokenizer, orig_emb, original_text, ind)
#             for ind in population
#         ]
        
#         for i, ind in enumerate(population):
#             if fitnesses[i] > best_fitness:
#                 best_fitness = fitnesses[i]
#                 best_individual = ind
        
#         sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
#         selection_probs = np.exp(-np.arange(pop_size) / (pop_size / 2))
#         selection_probs = selection_probs / selection_probs.sum()
        
#         new_pop = []
#         while len(new_pop) < pop_size:
#             idx1, idx2 = np.random.choice(
#                 sorted_indices, 
#                 size=2, 
#                 p=selection_probs,
#                 replace=False
#             )
#             p1, p2 = population[idx1], population[idx2]
            
#             c1, _ = crossover_5char(p1, p2)
#             c1 = mutation_5char(c1, char_table, mutation_rate)
#             new_pop.append(c1)
        
#         population = new_pop
    
#     best_sim = -best_fitness  # since fitness = -similarity
#     attacked_text = add_5_chars(original_text, best_individual)
#     return attacked_text, best_sim

# def genetic_5char_attack2(
#     original_text,
#     model,
#     tokenizer,
#     char_table,
#     pop_size=10,
#     generations=5,
#     mutation_rate=0.1
# ):
#     orig_emb = get_word_embeddings(model, tokenizer, original_text)
    
#     population = init_5char_population(char_table, pop_size)
#     best_individual = population[0]
#     best_fitness = fitness_function_5char(model, tokenizer, orig_emb, original_text, best_individual)
    
#     for _ in range(generations):
#         fitnesses = [
#             fitness_function_5char(model, tokenizer, orig_emb, original_text, ind)
#             for ind in population
#         ]
        
#         for i, ind in enumerate(population):
#             if fitnesses[i] > best_fitness:
#                 best_fitness = fitnesses[i]
#                 best_individual = ind
        
#         sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
#         top_k = sorted_pop[: len(sorted_pop)//2]
        
#         new_pop = []
#         while len(new_pop) < pop_size:
#             p1, _ = random.choice(top_k)
#             p2, _ = random.choice(top_k)
#             c1, c2 = crossover_5char(p1, p2)
#             c1 = mutation_5char(c1, char_table, mutation_rate)
#             c2 = mutation_5char(c2, char_table, mutation_rate)
#             new_pop.append(c1)
#             new_pop.append(c2)
#         population = new_pop[:pop_size]
    
#     best_attack_string = best_individual
#     best_sim = -best_fitness  # since fitness = -similarity
#     attacked_text = add_5_chars(original_text, best_attack_string)
#     return attacked_text, best_sim

# class PGD5CharAttacker:
#     def __init__(self, model, tokenizer, char_table, max_iter=20):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.char_table = char_table
#         self.max_iter = max_iter
    
#     def attack(self, original_text):
#         orig_emb = get_word_embeddings(self.model, self.tokenizer, original_text)
        
#         best_chars = [''] * 5
#         best_sim = 1.0
        
#         for _ in range(self.max_iter):
#             improved = False
#             for pos in range(5):
#                 current_char = best_chars[pos]
#                 local_best_char = current_char
#                 local_best_sim = best_sim
                
#                 for c in self.char_table:
#                     # Temporarily put c at pos
#                     best_chars[pos] = c
#                     # Check new similarity
#                     candidate_string = ''.join(best_chars)
#                     attacked_text = add_5_chars(original_text, candidate_string)
#                     new_emb = get_word_embeddings(self.model, self.tokenizer, attacked_text)
#                     sim_val = similarity(orig_emb, new_emb)
                    
#                     if sim_val < local_best_sim:
#                         local_best_sim = sim_val
#                         local_best_char = c
                
#                 best_chars[pos] = local_best_char
#                 if local_best_sim < best_sim:
#                     best_sim = local_best_sim
#                     improved = True
            
#             if not improved:
#                 break
        
#         best_string = ''.join(best_chars)
#         attacked_text = add_5_chars(original_text, best_string)
#         return attacked_text, best_sim
# def evaluate_attacks(
#     original_text: str,
#     attack_texts: dict,
#     model,
#     tokenizer,
#     id2label: dict,
# ):

#     original_preds = predict_ner_tags(
#         text=original_text,
#         model=model,
#         tokenizer=tokenizer,
#         id2label=id2label
#     )
#     orig_pred_tokens, orig_pred_tags = zip(*original_preds) if original_preds else ([], [])


#     results = {}
    
#     results["original"] = {
#         "precision": 1.0,
#         "recall": 1.0,
#         "f1": 1.0,
#         "accuracy": 1.0,
#     }
    
#     for attack_name, attacked_str in attack_texts.items():
#         attacked_preds = predict_ner_tags(
#             text=attacked_str,
#             model=model,
#             tokenizer=tokenizer,
#             id2label=id2label
#         )
        
#         att_pred_tokens, att_pred_tags = zip(*attacked_preds) if attacked_preds else ([], [])
        

#         len_orig = len(orig_pred_tags)
#         if len(att_pred_tags) < len_orig:
#             print(f"Warning: attacked prediction has fewer tokens ({len(att_pred_tags)}) than original ({len_orig}).")
        
#         att_pred_tags_short = att_pred_tags[:len_orig]
#         orig_pred_tags_short = orig_pred_tags[:len_orig]
        
#         # 3. Compute seqeval-based metrics
#         attack_metrics = compute_single_text_metrics(
#             list(att_pred_tags_short), 
#             list(orig_pred_tags_short)
#         )
#         results[attack_name] = attack_metrics
    
#     return results

# def compute_average_metrics(total_results, attack_names=None):
#     if attack_names is None:
#         attack_names = ["original", "greedy", "genetic", "pgd"]
    
#     metric_sums = {name: {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0} 
#                    for name in attack_names}
    
#     count = len(total_results)
    
#     for _, eval_results, in total_results:
#         for attack_name in attack_names:
#             for metric_name in ["precision", "recall", "f1", "accuracy"]:
#                 metric_sums[attack_name][metric_name] += eval_results[attack_name][metric_name]
                
#     for attack_name in attack_names:
#         for metric_name in ["precision", "recall", "f1", "accuracy"]:
#             metric_sums[attack_name][metric_name] /= count
    
#     return metric_sums
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from collections import Counter, defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def get_word_embeddings(model, tokenizer, sentence):
    model.eval()
    device = next(model.parameters()).device
    if isinstance(sentence, str):
        sentence = sentence.split()

    encoded = tokenizer(sentence, is_split_into_words=True, max_length=128, truncation=True, padding='max_length')
    input_ids = torch.tensor(encoded['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden_states = model(input_ids, attention_mask=attention_mask)
        # Take the output before the classification layer
        hidden_states = model.bert(input_ids, attention_mask=attention_mask)
        hidden_states = hidden_states.detach().cpu()[0]
    return hidden_states
def similarity(emb1, emb2):
    # Each emb is shape [seq_len, hidden_size]
    vec1 = emb1.mean(dim=0, keepdim=True)
    vec2 = emb2.mean(dim=0, keepdim=True)
    return cos(vec1, vec2).item()

def get_char_table():
    """
    A set of possible chars for simple 5-char attacks.
    """
    char_table = [
        '·','~','!','@','#','$','%','^','&','*','(',')','=','-','+',
        '.','<','>','?',',','\'',';',':','|','\\','/','_'
    ]
    for i in range(ord('A'), ord('Z')+1):
        char_table.append(chr(i))
    for i in range(ord('a'), ord('z')+1):
        char_table.append(chr(i))
    for i in range(10):
        char_table.append(str(i))
    return list(set(char_table))

def add_5_chars(original_text, attack_string):
    return original_text + " " + attack_string

def greedy_5char_attack(original_text, model, tokenizer, char_table):
    orig_emb = get_word_embeddings(model, tokenizer, original_text)
    best_sim = 1.0  # we want to push similarity down
    best_attack_chars = [''] * 5
    
    for pos in range(5):
        local_best_char = ''
        local_best_sim = best_sim
        
        for c in char_table:
            best_attack_chars[pos] = c
            candidate = ''.join(best_attack_chars)
            attacked_text = add_5_chars(original_text, candidate)
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
    return -sim_val  # negative => we want to minimize sim

def crossover_5char(parent1, parent2):
    idx = random.randint(1, 4)
    child1 = parent1[:idx] + parent2[idx:]
    child2 = parent2[:idx] + parent1[idx:]
    return child1, child2

def mutation_5char(child, char_table, mutation_rate=0.1):
    child_list = list(child)
    for i in range(len(child_list)):
        if random.random() < mutation_rate:
            child_list[i] = random.choice(char_table)
    return ''.join(child_list)

def genetic_5char_attack(original_text, model, tokenizer, char_table,
                         pop_size=10, generations=5, mutation_rate=0.1):
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
        
        # Sort descending
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        half = len(sorted_pop) // 2
        top_k = sorted_pop[:half]
        
        new_pop = []
        while len(new_pop) < pop_size:
            p1, _ = random.choice(top_k)
            p2, _ = random.choice(top_k)
            c1, c2 = crossover_5char(p1, p2)
            c1 = mutation_5char(c1, char_table, mutation_rate)
            c2 = mutation_5char(c2, char_table, mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        population = new_pop[:pop_size]

    best_sim = -best_fitness
    attacked_text = add_5_chars(original_text, best_individual)
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
                    best_chars[pos] = c
                    candidate = ''.join(best_chars)
                    attacked_text = add_5_chars(original_text, candidate)
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

######################################################
# 9. Simple Attack Evaluation
######################################################
def simple_predict_tags(text, model, tokenizer, idx2tag, max_length=128):
    """
    Run the model on a single text (string), and return (token, label) pairs.
    """
    # Convert text to tokens
    if isinstance(text, str):
        text_tokens = text.split()
    else:
        text_tokens = text  # already tokenized

    encoded = tokenizer(
        text_tokens,
        is_split_into_words=True,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    input_ids = torch.tensor(encoded["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(encoded["attention_mask"]).unsqueeze(0)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        
    # Just use outputs directly since it's the logits
    logits = outputs  # Changed this line
    pred_ids = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()

    # Map back to tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    # Reconstruct final (token, label) list
    final_pairs = []
    for token, label_id, mask_val in zip(tokens, pred_ids, encoded["attention_mask"]):
        if mask_val == 0:
            # padded
            break
        # Skip special tokens
        if token in [
            tokenizer.special_tokens['pad_token'],
            tokenizer.special_tokens['cls_token'],
            tokenizer.special_tokens['sep_token']
        ]:
            continue
        # Convert label ID to string
        label_str = idx2tag[label_id]
        # Handle subword prefix "##"
        if len(token) > 2 and token[:2] == "##":
            # Merge with previous
            if final_pairs:
                old_token, old_label = final_pairs[-1]
                if old_label == label_str:
                    merged_token = old_token + token[2:]
                    final_pairs[-1] = (merged_token, old_label)
                else:
                    # If subword has different predicted label, just store as new
                    final_pairs.append((token[2:], label_str))
            else:
                final_pairs.append((token[2:], label_str))
        else:
            final_pairs.append((token, label_str))
    return final_pairs
def get_entities_from_iob(seq):
    """
    Given a list of IOB tags (e.g., ["B-PER", "I-PER", "O", "B-ORG"]),
    return a list of (entity_type, start_index, end_index).
    Example output: [("PER", 0, 1), ("ORG", 3, 3)]
    """
    entities = []
    current_entity = None
    current_start = None
    current_label = None
    
    for i, tag in enumerate(seq):
        if tag == "O" or tag.startswith("B-"):
            # If we were tracking an entity, close it out
            if current_entity is not None:
                entities.append((current_entity, current_start, i - 1))
                current_entity = None
                current_start = None
            # Start of a new entity?
            if tag.startswith("B-"):
                current_entity = tag[2:]  # remove the "B-"
                current_start = i
                
        elif tag.startswith("I-"):
            # Continue the current entity if it matches the same type
            if current_entity is None:
                # This handles a corner case: "I-*" appears without a preceding "B-*"
                current_entity = tag[2:]
                current_start = i
            else:
                # Check if we switched to a different entity type incorrectly
                current_label = tag[2:]
                if current_label != current_entity:
                    # Close out the previous entity and start a new one
                    entities.append((current_entity, current_start, i - 1))
                    current_entity = current_label
                    current_start = i
        else:
            # Shouldn't happen with standard IOB, but just in case
            pass

    # If we ended in the middle of an entity
    if current_entity is not None:
        entities.append((current_entity, current_start, len(seq) - 1))
    
    return entities
def compute_chunk_metrics(true_list, pred_list):
    """
    Compute precision, recall, and F1 for chunk-level predictions.
    
    :param true_list: list of true tag sequences (e.g. [["B-PER", "I-PER", "O"], ...])
    :param pred_list: list of predicted tag sequences
    :return: a dictionary with overall precision, recall, and f1.
    """
    assert len(true_list) == len(pred_list), "Number of examples differs!"
    
    # For overall micro-average
    true_entities_all = 0
    pred_entities_all = 0
    match_entities_all = 0
    
    # For label-wise stats
    label_wise_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    for true_seq, pred_seq in zip(true_list, pred_list):
        true_entities = get_entities_from_iob(true_seq)
        pred_entities = get_entities_from_iob(pred_seq)
        
        # Convert to sets for direct matching
        # We'll store them as (type, start, end)
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        # For micro-averaging
        true_entities_all += len(true_set)
        pred_entities_all += len(pred_set)
        
        # Count matches
        matches = true_set.intersection(pred_set)
        match_entities_all += len(matches)
        
        # --- Label-wise stats ---
        # We'll track each entity in sets by label
        # A chunk is identified by the entity type.
        
        # Tally for each entity in "true_set"
        for ent in true_set:
            label = ent[0]
            # If it's not matched, it counts as FN
            if ent in matches:
                label_wise_stats[label]["TP"] += 1
            else:
                label_wise_stats[label]["FN"] += 1
        
        # Tally for each entity in "pred_set"
        for ent in pred_set:
            label = ent[0]
            # If it's not matched, it counts as FP
            if ent not in matches:
                label_wise_stats[label]["FP"] += 1

    # Compute overall micro P/R/F
    if pred_entities_all == 0:
        precision = 0.0
    else:
        precision = match_entities_all / pred_entities_all
    
    if true_entities_all == 0:
        recall = 0.0
    else:
        recall = match_entities_all / true_entities_all
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # Build a simple label-wise classification report
    classification_report_str = "label\tprecision\trecall\tf1\t\n"
    for label, stats in sorted(label_wise_stats.items()):
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]
        if (tp + fp) == 0:
            precision_label = 0.0
        else:
            precision_label = tp / (tp + fp)
        if (tp + fn) == 0:
            recall_label = 0.0
        else:
            recall_label = tp / (tp + fn)
        if precision_label + recall_label == 0:
            f1_label = 0.0
        else:
            f1_label = 2 * precision_label * recall_label / (precision_label + recall_label)

        classification_report_str += (
            f"{label}\t"
            f"{precision_label:.4f}\t"
            f"{recall_label:.4f}\t"
            f"{f1_label:.4f}\n"
        )

    # Add overall micro-average
    classification_report_str += (
        "\nOverall micro-average precision: {:.4f}\n".format(precision)
        + "Overall micro-average recall:    {:.4f}\n".format(recall)
        + "Overall micro-average f1:       {:.4f}\n".format(f1)
    )
    
    return classification_report_str, {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_single_text_metrics(predicted_tags, reference_tags):
    """
    Compare predicted_tags vs. reference_tags chunk-level.
    Both are lists of IOB labels of the same length.
    """
    report, summary = compute_chunk_metrics([reference_tags], [predicted_tags])
    # Return just the summary dictionary for convenience
    return summary

def evaluate_attacks(original_text, attack_texts, model, tokenizer, idx2tag):
    """
    - original_text: plain string (the reference).
    - attack_texts: dict {attack_name: attacked_string, ...}
    - We'll compare the chunk-based labels for each attacked text
      against the chunk-based labels of the original text's predictions.
    """
    # First, get model's predicted tags on the original text
    original_pairs = simple_predict_tags(original_text, model, tokenizer, idx2tag)
    if not original_pairs:
        return {}

    # Separate tokens vs. labels
    orig_tokens, orig_labels = zip(*original_pairs)

    # We'll treat the predicted labels on the original text as "ground truth"
    # (since we are measuring how the predicted labels shift after attack)
    results = {}
    results["original"] = dict(precision=1.0, recall=1.0, f1=1.0, accuracy=1.0)

    for attack_name, attacked_str in attack_texts.items():
        attacked_pairs = simple_predict_tags(attacked_str, model, tokenizer, idx2tag)
        if not attacked_pairs:
            results[attack_name] = dict(precision=0.0, recall=0.0, f1=0.0, accuracy=0.0)
            continue

        att_tokens, att_labels = zip(*attacked_pairs)

        # Truncate to length of original
        L = min(len(orig_labels), len(att_labels))
        orig_labels_short = orig_labels[:L]
        att_labels_short = att_labels[:L]

        summary = compute_single_text_metrics(list(att_labels_short), list(orig_labels_short))
        results[attack_name] = summary

    return results

def compute_average_metrics(total_results, attack_names=None):
    """
    total_results is a list of (original_text, eval_results) pairs.
    Each eval_results is what evaluate_attacks() returns.
    """
    if attack_names is None:
        attack_names = ["original", "greedy", "genetic", "pgd", "homoglyph"]

    metric_sums = {
        a: {"precision": 0.0, "recall": 0.0, "f1": 0.0,}
        for a in attack_names
    }
    count = len(total_results)

    for _, eval_res in total_results:
        for attack_name in attack_names:
            if attack_name not in eval_res:
                continue
            for m in ["precision", "recall", "f1", "accuracy"]:
                metric_sums[attack_name][m] += eval_res[attack_name][m]

    for attack_name in attack_names:
        for m in ["precision", "recall", "f1", "accuracy"]:
            metric_sums[attack_name][m] /= max(1, count)

    return metric_sums

######################################################
# 10. Homoglyph Attack Example
######################################################
def get_cyrillic_char_table():
    return {
        'A': 'А',  # Cyrillic capital A
        'a': 'а',  # Cyrillic lowercase a
        'B': 'В',
        'E': 'Е',
        'e': 'е',
        'I': 'І',
        'i': 'і',
        'K': 'К',
        'k': 'к',
        'M': 'М',
        'm': 'м',
        'H': 'Н',
        'O': 'О',
        'o': 'о',
        'P': 'Р',
        'p': 'р',
        'C': 'С',
        'c': 'с',
        'T': 'Т',
        't': 'т',
        'y': 'у',
        'X': 'Х',
        'x': 'х',
    }

def homoglyph_attack_5chars(text, char_table, model, tokenizer, num_replacements=5):
    """
    Randomly replaces up to `num_replacements` ASCII chars with visually similar
    Cyrillic chars. 
    """
    text_chars = list(text)
    replaceable_indices = [i for i, ch in enumerate(text_chars) if ch in char_table]

    random.shuffle(replaceable_indices)
    indices_to_replace = replaceable_indices[:num_replacements]

    for idx in indices_to_replace:
        ascii_char = text_chars[idx]
        text_chars[idx] = char_table[ascii_char]

    attacked_text = "".join(text_chars)
    orig_emb = get_word_embeddings(model, tokenizer, text)
    new_emb = get_word_embeddings(model, tokenizer, attacked_text)
    sim_val = similarity(orig_emb, new_emb)
    return attacked_text, sim_val