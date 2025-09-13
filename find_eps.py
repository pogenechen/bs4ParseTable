from .bs4ParseTable import bs4Parsed
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def find_eps_item(html):
    parsed = bs4Parsed(html, source='0000066570-20-000013')

    model = SentenceTransformer("bge-base-en-honsec10k-embed")
    sentence = 'basic earnings per share common shareholders'
    sentence_embedded = model.encode(sentence)
    sim_result = defaultdict(list)

    str_to_remove = r'\n| |\xa0|\\xa0|\u3000|\\u3000|\\u0020|\u0020|\t|\r'  

    max_sim = 0
    tb_idx = None
    tr_idx = None

    for tb in parsed.tables:
        for tr in tb.rows:
            tds = [re.sub(str_to_remove," ",td.text) for td in tr.cells]
            if not tds:
                continue
            td_embedded = model.encode(' '.join(tds))
            similarity = float(model.similarity(sentence_embedded, td_embedded))
            if similarity > max_sim:
                max_sim = similarity
                tb_idx = tb.idx
                tr_idx = tr.idx

    setattr(parsed[tb_idx][tr_idx],'similarity',similarity)
    setattr(parsed,'tb_idx',tb_idx)
    return parsed