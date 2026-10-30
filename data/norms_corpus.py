"""Корпус нормативных документов для поиска норм (RAG).

    python -m data.norms_corpus        # скачать и очистить тексты → docs/norms/<code>.txt

Источник — legalacts.ru: официальные тексты актов целиком на одной странице. Официальные документы
(законы, постановления, приказы) не охраняются авторским правом (п. 6 ст. 1259 ГК РФ), поэтому очищенные
тексты лежат в репозитории — оценка поиска воспроизводима, даже если страница источника изменится.
"""
import sys
from html.parser import HTMLParser
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "docs" / "norms"

DOCS = [
    {"code": "PTE511", "short": "ПТЭ объектов теплоснабжения (приказ Минэнерго № 511)",
     "title": "Приказ Минэнерго России от 14.05.2025 № 511 «Об утверждении Правил технической эксплуатации "
              "объектов теплоснабжения и теплопотребляющих установок»",
     "url": "https://legalacts.ru/doc/prikaz-minenergo-rossii-ot-14052025-n-511-ob-utverzhdenii/"},
    {"code": "PP354", "short": "Правила предоставления коммунальных услуг (ПП № 354)",
     "title": "Постановление Правительства РФ от 06.05.2011 № 354 «О предоставлении коммунальных услуг "
              "собственникам и пользователям помещений в многоквартирных домах и жилых домов»",
     "url": "https://legalacts.ru/doc/postanovlenie-pravitelstva-rf-ot-06052011-n-354/"},
    {"code": "PP1034", "short": "Правила коммерческого учёта тепловой энергии (ПП № 1034)",
     "title": "Постановление Правительства РФ от 18.11.2013 № 1034 «О коммерческом учете тепловой энергии, "
              "теплоносителя»",
     "url": "https://legalacts.ru/doc/postanovlenie-pravitelstva-rf-ot-18112013-n-1034/"},
    {"code": "SANPIN3684", "short": "СанПиН 2.1.3684-21",
     "title": "Постановление Главного государственного санитарного врача РФ от 28.01.2021 № 3 "
              "«Об утверждении санитарных правил и норм СанПиН 2.1.3684-21»",
     "url": "https://legalacts.ru/doc/postanovlenie-glavnogo-gosudarstvennogo-sanitarnogo-vracha-rf-ot-28012021-n/"},
]
BY_CODE = {d["code"]: d for d in DOCS}


class _Text(HTMLParser):
    """Абзацы документа (<p class="pBoth|pCenter|pRight">) и строки таблиц, по строке на элемент.

    Заголовки (pCenter, pRight — «Приложение N 1 к Правилам») помечаются «## », строки таблиц — ячейки через « | ».
    """

    def __init__(self):
        super().__init__()
        self.lines, self.buf = [], None
        self.kind = None                     # "p" | "h" | "cell"
        self.row, self.table_depth = None, 0

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "") or ""
        if tag == "table":
            self.table_depth += 1
        elif tag == "tr" and self.table_depth:
            self.row = []
        elif tag in ("td", "th") and self.table_depth:
            self.buf, self.kind = [], "cell"
        elif tag == "p" and cls.startswith("p") and not self.table_depth:
            self.buf, self.kind = [], "h" if cls in ("pCenter", "pRight") else "p"
        elif tag in ("br", "p") and self.buf is not None:      # абзацы внутри ячейки таблицы
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.kind == "cell" and self.buf is not None:
            self.row.append(" ".join("".join(self.buf).split()))
            self.buf = None
        elif tag == "tr" and self.row is not None:
            cells = [c for c in self.row if c]
            if cells:
                self.lines.append(" | ".join(cells))
            self.row = None
        elif tag == "table":
            self.table_depth -= 1
        elif tag == "p" and self.kind in ("p", "h") and self.buf is not None:
            text = " ".join("".join(self.buf).split())
            if text:
                self.lines.append(("## " if self.kind == "h" else "") + text)
            self.buf, self.kind = None, None

    def handle_data(self, data):
        if self.buf is not None:
            self.buf.append(data)


def fetch(doc):
    resp = requests.get(doc["url"], timeout=60, headers={"User-Agent": "itp-agent corpus builder"})
    resp.raise_for_status()
    parser = _Text()
    parser.feed(resp.text)
    return parser.lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        lines = fetch(doc)
        header = [f"# {doc['title']}", f"# Источник: {doc['url']}", ""]
        (OUT / f"{doc['code']}.txt").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
        print(f"{doc['code']}: {len(lines)} абзацев, {sum(map(len, lines)) // 1000} тыс. знаков")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
