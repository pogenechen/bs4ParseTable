import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
import outlines
import warnings
import torch

warnings.filterwarnings("ignore")

class ModelLoader:
    def __init__(
            self, 
            embedding_model_path='pavanmantha/bge-base-en-honsec10k-embed', 
            text_gen_model_path='next-tat/tat-llm-7b-fft', 
            use_outlines=False, 
            gpu_is_weak=False
            ):
        os.environ['TORCH_COMPILE_DISABLE'] = '1'
        os.environ['TORCHDYNAMO_DISABLE'] = '1'
        self.empty_cache()
        self.embedding_model_path = embedding_model_path
        self.text_gen_model_path = text_gen_model_path
        self.use_outlines = use_outlines
        self.gpu_is_weak = gpu_is_weak
        self.embedding_model = SentenceTransformer(self.embedding_model_path)
        self.text_gen_model, self.tokenizer = self.load_model()
        
    def load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.text_gen_model_path, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        if self.gpu_is_weak:
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
            self.text_gen_model_path,
            device_map="auto",
            torch_dtype="auto",
            use_safetensors=True,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        if self.use_outlines:
            model = outlines.from_transformers(model, tokenizer)
        else:
            model = pipeline("text-generation", model=model, tokenizer=tokenizer)

        return model, tokenizer
    
    def ask(self, prompt):

        if self.use_outlines:
            numeric_type = outlines.types.Regex(r"-?\d+\.\d*")
            ans = self.text_gen_model(prompt, numeric_type, max_new_tokens=8)
        else:
            ans = self.text_gen_model(
                prompt,
                max_new_tokens=8,
                do_sample=False,
                return_full_text=False,
                temperature=0,
                pad_token_id=self.tokenizer.pad_token_id
            )
            if ans:
                ans = ans[0]['generated_text'].strip()
        return ans

    def empty_cache(self):
        torch.cuda.empty_cache()

MODEL = ModelLoader(
    embedding_model_path=r"C:\Users\gene5\Desktop\Trexquant_--_DevOps_Engineer_Interview_Project__1_\bge-base-en-honsec10k-embed",
    text_gen_model_path=r"C:\Users\gene5\Desktop\Trexquant_--_DevOps_Engineer_Interview_Project__1_\tat-llm-7b-fft",
    use_outlines=False,
    gpu_is_weak=True
)