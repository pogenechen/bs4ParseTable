from models import MODEL
import re
from parser import parse_and_locate
import os
import pandas as pd
import warnings
from typing import List
import logging
warnings.filterwarnings("ignore")

def is_token_num_excessive(prompt, tokenizer,thresh=4000):
    tokens = tokenizer.encode(prompt)
    return True if len(tokens) > thresh else False

def extract_eps(
        htmls:List[str],
        model,
        prompt:str="Here is a markdown table: {markdown}\nQuestion: what is the latest GAAP basic earnings per share for shareholders?\nAnswer (just the number):",
        save_df:bool=True,
        **kwargs
)->pd.DataFrame:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="app.log",
        filemode="a"
    )
    results = {'file':[], 'ans':[]}
    logging.info(f'start extracting from  {len(htmls)}  files')
    for i, file in enumerate(htmls):
        try:
            file_name = os.path.basename(file)

            print(f"\rProcessing file {i+1}/{len(htmls)}: {file_name}",end='')

            with open(file,'r',encoding='utf8') as f:
                html = f.read()

            parsed = parse_and_locate(html, model=model.embedding_model, source_name=file_name)

            markdown = parsed[parsed.tb_idx].markdown

            if is_token_num_excessive(markdown, model.tokenizer):
                markdown = parsed.truncated_markdown

            prompt_ = prompt.format(markdown=markdown)

            ans = model.ask(
                prompt=prompt_
            )

            match_ = re.match(r"\-?\d+\.\d*",ans)
            if match_:
                ans = ans[slice(*match_.span())]


            results['filename'].append(file_name)
            results['EPS'].append(ans)

        except Exception as e:
            logging.error(f"failed to extract from {file_name},\nerror:{e}")

    df = pd.DataFrame(results)

    if save_df:
        save_to = kwargs.get('save_to','./results.csv')
        df.to_csv(save_to, index=False)
    logging.info('extraction finished')

    return df

if __name__ == "__main__":

    html_dir = input("Please input the directory path of HTML files: ").strip()
    save_to = input("Please input the path to save the CSV file (default: ./results.csv): ").strip()
    
    if not save_to:
        save_to = './results.csv'
    else:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)

    htmls = [os.path.join(html_dir, file) for file in os.listdir(html_dir) if file.endswith('htm') or file.endswith('html')]
    df = extract_eps(
        htmls=htmls,
        model=MODEL,
        save_df=True,
        save_to=save_to
    )