from bs4 import BeautifulSoup
from markdownify import markdownify as md
import mdpd


class TableData:
    def __init__(self, td):
        self.td = td
        self.text = td.get_text(strip=True)
        self.attrs = td.attrs

    def __repr__(self):
        return f"TableData(text={self.text!r}, attrs={self.attrs})"


class TableRow:
    def __init__(self, tr, idx=None):
        self.tr = tr
        self.idx = idx
        self.cells = [TableData(td) for td in tr.find_all(['td','th'])]

    def __getitem__(self, i):
        return self.cells[i]

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        return f"TableRow(idx={self.idx}, cells={self.cells})"


class Table:
    def __init__(self, bs4Table, **kwargs):
        self.table = bs4Table
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.rows = [TableRow(tr, idx=i) for i, tr in enumerate(bs4Table.find_all('tr'))]

    def __repr__(self):
        attrs = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        return f"Table({','.join(attrs)})"

    def __getitem__(self, i):
        return self.rows[i]

    def __len__(self):
        return len(self.rows)
    
    @property
    def markdown(self):
        return md(str(self.table))

    @property
    def df(self):
        _df = mdpd.from_md(self.markdown)
        _df = _df.replace({'':None}).dropna(how='all').dropna(how='all',axis=1)
        _df.columns = range(len(_df.columns))
        _df.index = range(len(_df))
        return _df
    
class bs4Parsed(BeautifulSoup):
    def __init__(self, markup, source, features='html.parser', **kwargs):
        self.features = features
        self.source = source
        self.kwargs = kwargs
        for k,v in kwargs.items():
            setattr(self, k, v)
        super().__init__(markup, features=features)
    
    def __repr__(self):
        attrs = [f"{k}={v!r}" for k,v in self.kwargs.items()]
        return f"bs4Parsed({','.join(attrs)})"
    
    def __getitem__(self,i):
        return self.tables[i]
    
    @property
    def tables(self):
        tbs = self.find_all('table')
        return [Table(table, source=self.source, idx=idx) for idx, table in enumerate(tbs)]