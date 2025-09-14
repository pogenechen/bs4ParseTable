from bs4ParseTable import bs4Parsed
import re


def find_eps_item(html, model, source_name):
    parsed = bs4Parsed(html, source=source_name)

    sentence = 'GAAP basic net income or earnings per share common for shareholders'
    sentence_embedded = model.encode(sentence)

    str_to_remove = r'\n| |\xa0|\\xa0|\u3000|\\u3000|\\u0020|\u0020|\t|\r'  

    max_sim = 0
    tb_idx = None
    tr_idx = None

    for tb in parsed.tables:
        for tr in tb.rows:
            for td in tr.cells:
                td.text = re.sub(str_to_remove,' ',td.text)
                if not td or td==' ':
                    continue
                td_embedded = model.encode(td.text)
                similarity = float(model.similarity(sentence_embedded, td_embedded))
                if similarity > max_sim:
                    max_sim = similarity
                    tb_idx = tb.idx
                    tr_idx = tr.idx
                    td_idx = td.idx
    setattr(parsed,'max_similarity',max_sim)
    setattr(parsed,'tb_idx',tb_idx)
    setattr(parsed,'tr_idx',tr_idx)
    setattr(parsed,'td_idx',td_idx)
    return parsed