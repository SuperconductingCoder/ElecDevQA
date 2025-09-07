from qa_metrics.RewardBert import RewardBert
from qa_metrics.f1 import f1_score_with_precision_recall

from tqdm import tqdm
import json
import re
from mlx_lm import load, generate
import lmstudio as lms
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def LLM_evaluator(model, tokenizer, question, reference, candidate):

    system_prompt = '''
        You are an expert evaluator. Compare the candidate answer to the reference answer and assign sub-scores.    
    '''
    user_prompt1=f"""
        Question:
        {question}

        Reference answer:
        {reference}

        Candidate answer:
        {candidate}

        Criteria:
        1. Accuracy (0–10): Is the information factually correct?
        2. Completeness (0–10): Does it include all important points from the reference answer?
        3. Clarity (0–10): Is the answer well-written and easy to understand?

    """
    user_prompt2="""  
        Return your evaluation in JSON format:  
        {"accuracy": X,
        "completeness": X,
        "clarity": X
        }
    """
    user_prompt = user_prompt1+user_prompt2
    
    if tokenizer.chat_template is not None:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    for i in range(3):
        response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=1_0000)
        # print(response)
        try:
            json_pattern = r'\{.*\}'
            LLM_eval = re.search(json_pattern, response, re.DOTALL).group(0)
            LLM_eval = json.loads(LLM_eval)
            return LLM_eval
        except:
            print("JSON format error")
            print(response)
            LLM_eval={}
    return LLM_eval

class BERT_evaluator:
    def __init__(self, model, tokenizer, candidate, reference):
        cand_tokenized = tokenizer(candidate, padding=True, truncation=True, return_tensors='pt')
        ref_tokenized = tokenizer(reference, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            self.can_encode = model(**cand_tokenized)
            self.ref_encode = model(**ref_tokenized)
    
    def recall_BERT(self ):#Recall BERT
        sum = 0    
        ref_token_length = len(self.ref_encode["last_hidden_state"][0])
        for ref_vector in self.ref_encode["last_hidden_state"][0]:
            inner_product_list = []
            for gen_vector in self.can_encode["last_hidden_state"][0]:
                inner_product= F.cosine_similarity(ref_vector, gen_vector, dim=0).item()
                inner_product_list.append(inner_product)
            most_similar = max(inner_product_list)
            sum+=most_similar
        R_BERT = sum/ref_token_length
        return R_BERT

    
    def precision_BERT(self):#Precision BERT
        sum = 0        
        gen_token_length = len(self.can_encode["last_hidden_state"][0])
            
        for gen_vector in self.can_encode["last_hidden_state"][0]:
            inner_product_list = []
            for ref_vector in self.ref_encode["last_hidden_state"][0]:
                inner_product= F.cosine_similarity(ref_vector, gen_vector, dim=0).item()
                inner_product_list.append(inner_product)
            most_similar = max(inner_product_list)
            sum+=most_similar
        P_BERT = sum/gen_token_length
        return P_BERT

    def f1_BERT(self):   
        inner_product_arrays = []
        for ref_vector in self.ref_encode["last_hidden_state"][0]:
            ref_inner_product_list = []
            for gen_vector in self.can_encode["last_hidden_state"][0]:
                inner_product= F.cosine_similarity(ref_vector, gen_vector, dim=0).item()
                ref_inner_product_list.append(inner_product)
            inner_product_arrays.append(ref_inner_product_list) 
        num_rows = len(inner_product_arrays)
        num_cols = len(inner_product_arrays[0])
        row_maxes = []
        for row in inner_product_arrays:
            row_max = max(row)
            row_maxes.append(row_max)
        R_BERT = sum(row_maxes)/num_rows
        
        column_maxes = []
        for j in range(num_cols):
            column_elements = [inner_product_arrays[i][j] for i in range(num_rows)]
            column_max = max(column_elements)
            column_maxes.append(column_max)
        P_BERT = sum(column_maxes)/num_cols
        F_BERT = 2*(R_BERT * P_BERT)/(R_BERT + P_BERT)
        return R_BERT, P_BERT, F_BERT

class full_eval:
    def __init__(self, Q_candA_list):
        self.Q_candA_list= Q_candA_list
        self.num_quest= len(Q_candA_list)
            
    def f1_full_eval(self, output_folder):
        output_path = f"../QA_data/{output_folder}/f1.json"
        print("----f1 evaluation--")
        print(f"----output file: {output_path}")
        f1_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}    
        R_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}
        P_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}
        
        f1_eval=[]
        for Q_candA in tqdm(self.Q_candA_list):
            # print(Q_candA)
            id = Q_candA["id"]
            question = Q_candA["question"]
            new_question = Q_candA["new_question"]
            reference = Q_candA["reference"]
            candidates = Q_candA["candidates"]
            f1_scores={}
            for cand in candidates:
                answer = candidates[cand]
                f1_score = f1_score_with_precision_recall(reference, answer)
                f1_scores[cand] = f1_score
                #for average of all the scores
                f1_avg[cand] += f1_score["f1"]
                P_avg[cand] += f1_score["precision"]
                R_avg[cand] += f1_score["recall"]
                
            f1_eval.append({"id": id, "question":question, "new_question":new_question, "f1_scores":f1_scores})
        f1_avg = {key: value / self.num_quest for key, value in f1_avg.items()}
        P_avg = {key: value / self.num_quest for key, value in P_avg.items()}
        R_avg = {key: value / self.num_quest for key, value in R_avg.items()}
        output_dict = {"f1_avg":f1_avg, "R_avg":R_avg, "P_avg":P_avg, "f1_eval":f1_eval}
        with open(output_path, 'w') as output_file:
            json.dump(output_dict, output_file, indent=4)
      
    def rb_full_eval(self, output_folder):
        output_path = f"../QA_data/{output_folder}/rb.json"
        print("----rewardBert evaluation--")
        print(f"----output file: {output_path}")
        rb_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0} 

        rb = RewardBert()

        rb_eval=[]
        for Q_candA in tqdm(self.Q_candA_list):
            id = Q_candA["id"]
            question = Q_candA["question"]
            new_question = Q_candA["new_question"]
            reference = Q_candA["reference"]
            candidates = Q_candA["candidates"]
            rb_scores={}
            for cand in candidates:
                answer = candidates[cand]
                score = rb.compute_score(reference, answer)
                rb_scores[cand] = score[0]
                rb_avg[cand]+=score[0]
            rb_eval.append({"id": id, "question":question, "new_question":new_question, "rb_scores":rb_scores})
        rb_avg = {key: value / self.num_quest for key, value in rb_avg.items()}
        output_dict = {"rb_avg":rb_avg, "rb_eval":rb_eval}
        with open(output_path, 'w') as output_file:
            json.dump(output_dict, output_file, indent=4)

    def BERT_full_eval(self, output_folder):
        output_path = f"../QA_data/{output_folder}/BertRecall.json"
        print("----BertRecall evaluation--")
        print(f"----output file: {output_path}")
        f1_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}    
        R_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}
        P_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}
        
        model_name = "sentence-transformers/all-roberta-large-v1"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)    
        
        BertRecall_eval=[]
        for Q_candA in tqdm(self.Q_candA_list):
            id = Q_candA["id"]
            question = Q_candA["question"]
            new_question = Q_candA["new_question"]
            reference = Q_candA["reference"]
            candidates = Q_candA["candidates"]
            BertRecall_scores={}
            for cand in candidates:
                answer = candidates[cand]
                evaluator = BERT_evaluator(model, tokenizer, answer, reference)
                recall, precision, f1 =  evaluator.f1_BERT()
                BertRecall_scores[cand] = {"f1":f1, "precision":precision, "recall":recall}
                
                f1_avg[cand] += f1
                P_avg[cand] += precision
                R_avg[cand] += recall
            BertRecall_eval.append({"id": id, "question":question, "new_question":new_question, "BertRecall_scores":BertRecall_scores})
        f1_avg = {key: value / self.num_quest for key, value in f1_avg.items()}
        P_avg = {key: value / self.num_quest for key, value in P_avg.items()}
        R_avg = {key: value / self.num_quest for key, value in R_avg.items()}
        output_dict = {"f1_avg":f1_avg, "R_avg":R_avg, "P_avg":P_avg, "BertRecall_eval":BertRecall_eval}
        with open(output_path, 'w') as output_file:
            json.dump(output_dict, output_file, indent=4)

    def gemma3_full_eval(self, output_folder):
        output_path = f"../QA_data/{output_folder}/gemma3.json"
        print("----LLM Gemma 3 evaluation--")
        print(f"----output file: {output_path}")
        try:
            with open(output_path, "r") as f:
                processedData = json.load(f)
            gemma3_eval = processedData["gemma3_eval"]
        except:
            gemma3_eval=[]
        
        model, tokenizer = load("/Users/borwoeihuang/.lmstudio/models/mlx-community/gemma-3-12b-it-qat-4bit")
        accuracy_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}    
        completeness_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}
        clarity_avg={"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}
        total_avg = {"qwen_ans":0, "llama31_ans":0, "phi4_ans":0, "gemma3_ans":0, "GPToos_ans":0}

        num_processed = len(gemma3_eval)
        for i, Q_candA in tqdm(enumerate(self.Q_candA_list)):
            if (i+1)>num_processed:
                id = Q_candA["id"]
                question = Q_candA["question"]
                new_question = Q_candA["new_question"]
                reference = Q_candA["reference"]
                candidates = Q_candA["candidates"]
                gemma3_scores={}
                for cand in candidates:
                    answer = candidates[cand]
                    scores = LLM_evaluator(model, tokenizer, new_question, reference, answer)
                    if scores!={}:
                        gemma3_scores[cand] = scores
                        
                        accuracy= scores["accuracy"]/10
                        completeness= scores["completeness"]/10
                        clarity= scores["clarity"]/10
                        total_score = (accuracy + completeness + clarity)/3
                        accuracy_avg[cand]+=accuracy
                        completeness_avg[cand]+=completeness
                        clarity_avg[cand]+=clarity
                        total_avg[cand]+=total_score
                    
                gemma3_eval.append({"id": id, "question":question, "new_question":new_question, "gemma3_scores":gemma3_scores})
                output_dict = {"gemma3_eval":gemma3_eval}
                with open(output_path, 'w') as output_file:
                    json.dump(output_dict, output_file, indent=4)
                
        accuracy_avg = {key: value / self.num_quest for key, value in accuracy_avg.items()}
        completeness_avg = {key: value / self.num_quest for key, value in completeness_avg.items()}
        clarity_avg = {key: value / self.num_quest for key, value in clarity_avg.items()}
        total_avg = {key: value / self.num_quest for key, value in total_avg.items()}
        output_dict = {"total_avg":total_avg,"accuracy_avg":accuracy_avg, "completeness_avg":completeness_avg, "clarity_avg":clarity_avg, "gemma3_eval":gemma3_eval}
        with open(output_path, 'w') as output_file:
            json.dump(output_dict, output_file, indent=4)
       
def candidate_ans_eval():
    Q_candA_list=[]
    files = ["1", "2", "3"]
    for file in files:
        read_path = f"../QA_data/QA_{file}_eval.jsonl"

        with open(read_path, 'r') as readfile:
            for line in readfile:
                data = json.loads(line)
                id = data["id"]
                question = data["question"]
                if data["new_QA"]!={}:
                    new_question = data["new_QA"]["question"]
                    reference = data["new_QA"]["answer"]
                    qwen_ans = data["candidates"]["qwen_ans"]
                    llama31_ans = data["candidates"]["llama31_ans"]
                    phi4_ans = data["candidates"]["phi4_ans"]
                    gemma3_ans = data["candidates"]["gemma3_ans"]
                    GPToos_ans = data["candidates"]["GPToos_ans"]
                    candidates = {"qwen_ans":qwen_ans, "llama31_ans":llama31_ans, "phi4_ans":phi4_ans, "gemma3_ans":gemma3_ans, "GPToos_ans":GPToos_ans}

                    Q_candA_list.append({"id":id, "question":question, "new_question":new_question, "reference":reference, "candidates":candidates})
        evaluator = full_eval(Q_candA_list)
        output_folder =f"file{file}_cand5"
        evaluator.f1_full_eval(output_folder)
        evaluator.rb_full_eval(output_folder)
        evaluator.BERT_full_eval(output_folder)
        evaluator.gemma3_full_eval(output_folder) 

if __name__ == "__main__":
    candidate_ans_eval()
