from bs4ParseTable import bs4Parsed
import re
from bs4 import BeautifulSoup
from markdownify import markdownify as md


def parse_and_locate(html, model, source_name, sentence = 'GAAP basic net income or earnings per share common for shareholders'):
    parsed = bs4Parsed(html, source=source_name)

    sentence = 'basic earnings per share for common shareholders'
    sentence_embedded = model.encode(sentence)

    max_sim = 0
    tb_idx = None
    tr_idx = None
    all_sim = []
    for tb in parsed.tables:
        for tr in tb.rows:
            td_text_embedded = model.encode([td.text for td in tr.cells])
            similarities = [float(model.similarity(td_embedded, sentence_embedded)) for td_embedded in td_text_embedded]
            similarity = max(similarities)
            all_sim.append(((tb.idx, tr.idx), similarity))

            if similarity > max_sim:
                max_sim = similarity
                tb_idx = tb.idx
                tr_idx = tr.idx
                td_idx = similarities.index(similarity)

    top_2 = [i[0] for i in sorted(all_sim, key=lambda x:x[1], reverse=True)[:2]]
    soup = BeautifulSoup()
    truncated_table = soup.new_tag('table')
    for tb_i, tr_i in top_2:
        if tr_i>=1 and tr_i <= len(parsed[tb_i].rows)-2:
            for i in range(tr_i-1, tr_i+2):
                tr_clone = BeautifulSoup(str(parsed[tb_i][i].tr), "html.parser")
                truncated_table.append(tr_clone)
        else:
            tr_clone = BeautifulSoup(str(parsed[tb_i][tr_i].tr), "html.parser")
            truncated_table.append(tr_clone)

    pattern = r"\((\d+(?:\.\d+)?)\)?"
    truncated_markdown = re.sub(pattern, r'-\1', md(str(truncated_table)))
    setattr(parsed, 'all_sim', all_sim)
    setattr(parsed,'max_similarity',max_sim)
    setattr(parsed,'tb_idx',tb_idx)
    setattr(parsed,'tr_idx',tr_idx)
    setattr(parsed,'td_idx',td_idx)
    setattr(parsed,'truncated_markdown', truncated_markdown)

    return parsed