from find_eps import find_eps_item
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import outlines
from sentence_transformers import SentenceTransformer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path, use_outlines=True, gpu_is_weak=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if gpu_is_weak:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            llm_int8_enable_fp32_cpu_offload=True
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    if use_outlines:
        model = outlines.from_transformers(model, tokenizer)
    else:
        model = pipeline("text-generation", model=model, tokenizer=tokenizer)

    return model, tokenizer


def truncate_prompt(prompt, tokenizer, max_len):
    tokens = tokenizer(prompt, truncation=True, max_length=max_len)
    return tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)


def ask(model, tokenizer, markdown, question, use_outlines=True):
    prompt = f"Here is a markdown table: {markdown}\nQuestion: {question}\nAnswer (just the number):"

    max_len = getattr(model.model.config, "max_position_embeddings", 2048)
    prompt = truncate_prompt(prompt, tokenizer, max_len)

    if use_outlines:
        numeric_type = outlines.types.Regex(r"-?\d+\.\d*")
        ans = model(prompt, numeric_type, max_new_tokens=8)
    else:
        ans = model(
            prompt,
            max_new_tokens=8,
            do_sample=False,
            return_full_text=False,
            temperature=0,
            pad_token_id=tokenizer.pad_token_id
        )
        if ans:
            ans = ans[0]['generated_text'].strip()
    return ans


if __name__ == "__main__":
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    use_outlines = True

    html_dir = r'C:\Users\gene5\Desktop\Trexquant_--_DevOps_Engineer_Interview_Project__1_\Training_Filings'
    htmls = [os.path.join(html_dir,i) for i in os.listdir(html_dir) if i.endswith(('htm','html'))]
    results = {'file':[], 'ans':[]}
    
    embedding_model = SentenceTransformer(
        r'C:\Users\gene5\Desktop\Trexquant_--_DevOps_Engineer_Interview_Project__1_\bge-base-en-honsec10k-embed'
    )
    text_gen_model, tokenizer = load_model(
        r'C:\Users\gene5\Desktop\Trexquant_--_DevOps_Engineer_Interview_Project__1_\tat-llm-7b-fft',
        use_outlines=use_outlines,
        gpu_is_weak=True
    )

    for i, file in enumerate(htmls):
        try:
            file_name = os.path.basename(file).split('.')[0]
            print(f"Processing file {i+1}/{len(htmls)}: {file_name}", end='\r')
            with open(file,'r',encoding='utf8') as f:
                html = f.read()
            parsed = find_eps_item(
                html,
                model=embedding_model,
                source_name=file_name
            )

            target_item = parsed[parsed.tb_idx][parsed.tr_idx][parsed.td_idx].text
            markdown = parsed[parsed.tb_idx].markdown
            question = "what is the latest GAAP basic earnings per share for shareholders?"

            ans = ask(
                text_gen_model,
                tokenizer,
                markdown=markdown,
                question=question,
                use_outlines=use_outlines
            )
            results['file'].append(file_name)
            results['ans'].append(ans)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    df = pd.DataFrame(results)
    df.to_csv('outlines_results.csv', index=False)
