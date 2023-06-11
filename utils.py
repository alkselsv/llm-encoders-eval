import numpy as np
import torch


def embed_bert_cls(text, model, tokenizer, max_length=128):
    t = tokenizer(text, padding=True, truncation=True,
                  max_length=max_length, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def mean_pooling(model_output, attention_mask, norm=False):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    sums = sum_embeddings / sum_mask
    if norm:
        sums = torch.nn.functional.normalize(sums)
    return sums


def embed_bert_pool(text, model, tokenizer, max_length=128, norm=False):
    encoded_input = tokenizer(
        text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    if norm:
        embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def embed_bert_both(text, model, tokenizer, max_length=128):
    t = tokenizer(text, padding=True, truncation=True,
                  max_length=max_length, return_tensors='pt')
    t = {k: v.to(model.device) for k, v in t.items()}
    with torch.no_grad():
        model_output = model(**t)
    cls_emb = model_output.last_hidden_state[:, 0, :]
    cls_emb_norm = torch.nn.functional.normalize(cls_emb)
    mean_emb = mean_pooling(model_output, t['attention_mask'])
    mean_emb_norm = torch.nn.functional.normalize(mean_emb)

    return np.concatenate((cls_emb[0].cpu().numpy(), mean_emb[0].cpu().numpy()))


def find_most_similar_vector_indices(vector_array):
    normalized_vectors = vector_array / \
        np.linalg.norm(vector_array, axis=1)[:, np.newaxis]
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    np.fill_diagonal(similarity_matrix, -1)

    most_similar_indices = np.argmax(similarity_matrix, axis=1)

    return most_similar_indices
