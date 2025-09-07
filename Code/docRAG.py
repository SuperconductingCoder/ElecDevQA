from pyserini.search.lucene import LuceneSearcher
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def bm25_search(GS):
    index_path= "/index"
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(0.5, 0.3)
    hits = searcher.search(GS, k=20)
    doc_list = [hit.docid for hit in hits]
    return doc_list

def BERT_rerank(pairs, threshold):
    model_name_or_path = "Alibaba-NLP/gte-reranker-modernbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path,torch_dtype=torch.float16,)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=1024)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)
    doc_selected = []
    for i, score in enumerate(scores):
        if score > threshold:
            doc_selected.append(pairs[i][1])
    return doc_selected

if __name__ == "__main__":
    for question in question_list:
        doc_list = bm25_search(GS)
        pairs = [[question, GS_dict[doc_id]] for doc_id in doc_list]
        threshold = 0.53
        doc_selected = BERT_rerank(pairs, threshold)