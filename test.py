from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import outlines
from transformers import pipeline

def extract_with_outlines(tb):
    model_id = "tat-llm-7b-fft"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, 
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     load_in_4bit=True
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,   
    #     bnb_4bit_use_double_quant=True,         
    #     bnb_4bit_quant_type="nf4"               
    # )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  
        load_in_8bit=True,  
        llm_int8_enable_fp32_cpu_offload=True  
    )

    torch.cuda.empty_cache()
    outlines_model = outlines.from_transformers(model, tokenizer)

    numeric_type = outlines.types.Regex(r"-?\d+(\.\d+)?")

    question = "what is the latest unadjusted basic earnings per share"
    prompt = f"find me the latest unadjusted basic earnings per share from the markdown table:{tb.markdown}, return a numeric value"
    outlines_model(prompt, numeric_type)


def extract_with_pipeline(tb):
    pipe = pipeline(
        "text-generation",
        model="tat-llm-7b-fft",
        device_map="auto",             
        torch_dtype="auto",            
        model_kwargs={"load_in_4bit": True}  
    )

    question = "what is the latest unadjusted basic earnings per share?"
    prompt = f"Here is a markdown table: {tb.markdown}\nQuestion: {question}\nAnswer (just the numeric value):"

    ans = pipe(prompt, max_new_tokens=50, do_sample=False)
    return ans