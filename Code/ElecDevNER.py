import json
from tqdm import tqdm
import ollama

def create_message(message, role):
    return {
        'role': role,
        'content': message
    }

def chat():
    ollama_response = ollama.chat(model='llama3.1', stream=True, messages=chat_messages)
    assistant_message = ''
    for chunk in ollama_response:
        assistant_message += chunk['message']['content']
    chat_messages.append(create_message(assistant_message, 'assistant'))
    return assistant_message

def ask(message):
    chat_messages.append(
        create_message(message, 'user')
    )
    answer = chat()
    return answer

def llama_answer(abstract):
    global chat_messages
    chat_messages = []
    
    prompt0 = f"Given {abstract}. List the electronic devices and systems in the text. If none, print None"
    prompt1 = f"List the electronic materials of the electronic devices and systems in the text. If none, print None" 
    prompt2 = f"List the functions and features of the electronic devices"
    prompt3 = f"List the properties, numerical values of the electronic devices, and describe their meaning and unit. If no, return None."

    ans0 = ask(prompt0)
    ans1 = ask(prompt1)
    ans2 = ask(prompt2)
    ans3 = ask(prompt3)
    llama_answer = {"llama_rewrite": ans0, "electronic device": ans1, "functions":ans2, "feature":ans2, "property":ans3}
    
    return llama_answer

if __name__ == "__main__":
    

    for file in range(0,10):
        read_path = f"../IEEE_abstracts/IEEE_abstracts_{file}.jsonl"
        output_path = f"../IEEE_abstracts/IEEE_abstracts_{file}_Extract.jsonl"

        with open(read_path, 'r') as readfile:
            num_readfile_lines = sum(1 for line in readfile)
            
        try:
            with open(output_path, 'r') as outfile:
                num_outfile_lines = sum(1 for line in outfile)
        except:
            num_outfile_lines=0
            
        with open(read_path, 'r') as read_file:  
            for i, line in enumerate(tqdm(read_file, total=num_readfile_lines)):
                if (i+1)> num_outfile_lines:
                    scraped_info = json.loads(line)
                    if "llama_output" in scraped_info:
                        with open(output_path, 'a') as output_file:
                            json.dump(scraped_info, output_file)
                            output_file.write('\n')
                    else:
                        abstract = scraped_info["Abstract"]
                        llama_output = llama_answer( abstract)
                        scraped_info["llama_output"] = llama_output
                        with open(output_path, 'a') as output_file:
                            json.dump(scraped_info, output_file)
                            output_file.write('\n')
