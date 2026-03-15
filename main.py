import json
import re
import time

import spacy
import glirel
from gliner import GLiNER
import yfinance as yf

import spacy
from spacy.language import Language
from gliner import GLiNER
import glirel

import json
import os
import time
import requests
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import re

from concurrent.futures import ThreadPoolExecutor, as_completed

KNOWN_ACRONYMS = {
    "oem", "oems", "qct", "mdm", "gdp", "ceo", "cfo", "coo", "cto",
    "ip", "ai", "ml", "iot", "5g", "4g", "3g", "lte", "rf", "soc",
    "cpu", "gpu", "dsp", "api", "sdk", "usb", "led", "lcd", "ram",
    "esg", "sec", "irs", "doj", "ftc", "fcc", "gaap", "r&d",
    "m&a", "ipo", "eps", "roi", "p&l", "saas", "paas", "iaas",
}

@Language.factory("gliner_custom", default_config={
    "model_path": "./models/NuNER_Zero",
    "labels": ["company", "organization"],
    "threshold": 0.4,
    "chunk_size": 250,
})
class GLiNERCustom:
    def __init__(self, nlp, name, model_path, labels, threshold, chunk_size):
        self.model = GLiNER.from_pretrained(model_path)
        self.labels = labels
        self.threshold = threshold
        self.chunk_size = chunk_size

    ENTITY_BLACKLIST = {
        "business", "customers", "competitors", "governments", "government",
        "companies", "industry", "manufacturers", "suppliers", "partners",
        "clients", "vendors", "regulators", "shareholders", "investors",
        "consumers", "users", "mobile industry", "semiconductor business",
        "risk factor", "factor", "chinese governments", "device share companies",
        "oems", "u.s.", "chinese original equipment manufacturers",
        "systems", "manufacturing", "operations", "products", "services",
        "technology", "technologies", "devices", "revenues", "markets",
    }

    def __call__(self, doc):
        entities = self.model.predict_entities(
            doc.text, self.labels, threshold=self.threshold
        )
        entities = self._merge_entities(entities, doc.text)

        entities = [e for e in entities if e["text"].lower().strip() not in self.ENTITY_BLACKLIST]

        spans = []
        for ent in entities:
            span = doc.char_span(ent["start"], ent["end"], label=ent["label"])
            if span is not None:
                spans.append(span)
        doc.ents = spacy.util.filter_spans(spans)
        return doc

    @staticmethod
    def _merge_entities(entities, text):
        if not entities:
            return []
        merged = []
        current = entities[0].copy()
        for next_ent in entities[1:]:
            if (next_ent["label"] == current["label"]
                    and next_ent["start"] - current["end"] <= 1):
                current["text"] = text[current["start"]:next_ent["end"]].strip()
                current["end"] = next_ent["end"]
            else:
                merged.append(current)
                current = next_ent.copy()
        merged.append(current)
        return merged

def process_chunk(chunk, nlp, labels):
    """Process a single chunk through GLiREL."""
    docs = list(nlp.pipe([(chunk, labels)], as_tuples=True))
    doc = docs[0][0]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relations = doc._.relations
    return entities, relations


def process_all_chunks(chunks, nlp, labels, max_workers=10):
    """Process all chunks in parallel."""
    all_relations = []
    all_entities = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk, nlp, labels): chunk
            for chunk in chunks
        }

        for future in as_completed(futures):
            entities, relations = future.result()
            all_entities.extend(entities)
            all_relations.extend(relations)

    return all_entities, all_relations

def get_ticker(text):
    if text.lower().strip() in KNOWN_ACRONYMS:
        return None
    search = yf.Search(query=text, max_results=1).search()
    ls = list(search._all.values())[0]
    if len(ls) == 0:
        return None
    return ls[0]["symbol"]

GENERIC_BUSINESS_TERMS = {
    "oems", "customers", "competitors", "governments", "government",
    "companies", "business", "industry", "manufacturers", "suppliers",
    "subcontractors", "licensees", "partners", "clients", "vendors",
    "regulators", "shareholders", "investors", "consumers", "users",
    "mobile industry", "semiconductor business", "risk factor",
    "chinese governments", "device share companies",
}


def clean_filing_text(text):
    """Remove SEC filing artifacts that aren't real content."""
    # Page markers: F-18, F-1, A-3, S-12, etc.
    text = re.sub(r'\b[A-Z]-\d{1,3}\b', '', text)

    # Standalone page numbers
    text = re.sub(r'(?<=\n)\s*\d{1,3}\s*(?=\n)', '', text)

    # Table of contents markers
    text = re.sub(r'(?i)table of contents', '', text)

    # SEC header boilerplate
    text = re.sub(r'(?i)(UNITED STATES SECURITIES AND EXCHANGE COMMISSION|'
                  r'Washington,?\s*D\.?C\.?\s*\d{5}|'
                  r'FORM 10-K|'
                  r'ANNUAL REPORT PURSUANT TO SECTION)', '', text)

    # Repeated dashes or underscores used as dividers
    text = re.sub(r'[-_]{3,}', '', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def is_likely_entity_name(text):
    """Quick check before expensive API call."""
    lower = text.lower().strip()

    if lower in GENERIC_BUSINESS_TERMS:
        return False

    if len(lower.split()) == 1 and lower.islower():
        return False

    generic_suffixes = {"customers", "companies", "governments", "manufacturers",
                        "competitors", "suppliers", "partners", "devices",
                        "products", "services", "markets", "industries"}
    last_word = lower.split()[-1]
    if last_word in generic_suffixes:
        return False

    return True

def get_ticker_safe(text):
    lower = text.lower().strip()

    if not is_likely_entity_name(text):
        return None

    if lower in KNOWN_ACRONYMS:
        return None

    stripped = text.strip()
    if stripped.isupper() and len(stripped) <= 5 and " " not in stripped:
        # Only accept if Yahoo returns an EXACT ticker match
        try:
            search = yf.Search(query=text, max_results=1).search()
            ls = list(search._all.values())[0]
            if len(ls) == 0:
                return None
            result = ls[0]
            # Must be an exact ticker match, not a fuzzy name match
            if result["symbol"].upper() == stripped.upper():
                return result["symbol"]
            return None
        except Exception:
            return None

    # For longer names, use the word overlap approach
    try:
        search = yf.Search(query=text, max_results=1).search()
        ls = list(search._all.values())[0]
        if len(ls) == 0:
            return None

        result = ls[0]
        name = result.get("longname", result.get("shortname", "")).lower()
        query_words = {w for w in lower.split() if len(w) > 2}
        name_words = {w for w in name.split() if len(w) > 2}

        if query_words & name_words:
            return result["symbol"]

        return None
    except Exception:
        return None

def filter_chunks(chunks, nlp):
    valid = []
    for chunk in chunks:
        doc = nlp.make_doc(chunk)
        for name, proc in nlp.pipeline:
            if name == "glirel":
                break
            doc = proc(doc)
        unique_ents = {ent.text.lower() for ent in doc.ents}
        if len(unique_ents) >= 2:
            valid.append(chunk)
    return valid

def read_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def resolve_company_references(text, ticker):
    nlp_ref = spacy.blank("en")
    doc = nlp_ref(text.strip())

    replacements = {
        "we": ticker,
        "our": f"{ticker}'s",
        "us": ticker,
        "ourselves": ticker,
        "the company": ticker,
        "the company's": f"{ticker}'s",
    }

    tokens = []
    i = 0
    while i < len(doc):
        if i < len(doc) - 1:
            bigram = doc[i].text.lower() + " " + doc[i + 1].text.lower()
            if bigram in replacements:
                tokens.append(replacements[bigram])
                i += 2
                continue

        lower = doc[i].text.lower()
        if lower in replacements:
            replacement = replacements[lower]
            tokens.append(replacement)
        else:
            tokens.append(doc[i].text)
        i += 1

    return " ".join(tokens)

def smart_chunk(text, max_sentences=2):
    """Chunk text keeping entity-linked sentences together."""
    sentencizer = spacy.blank("en")
    sentencizer.add_pipe("sentencizer")
    doc = sentencizer(text.strip())
    sentences = list(doc.sents)

    if len(sentences) <= max_sentences:
        return [text.strip()]

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sent = sentences[i]
        at_limit = len(current_chunk) >= max_sentences

        first_token = sent[0].lower_ if len(sent) > 0 else ""
        linking_words = {
            "additionally", "consequently", "furthermore", "moreover",
            "however", "therefore", "thus", "also", "meanwhile",
            "similarly", "likewise", "accordingly", "hence",
        }
        is_linked = first_token in linking_words

        if is_linked and not at_limit:
            current_chunk.append(sent)
        else:
            chunks.append(" ".join(s.text.strip() for s in current_chunk))
            current_chunk = [sent]

    if current_chunk:
        chunks.append(" ".join(s.text.strip() for s in current_chunk))

    return chunks

def filter_real_companies(relations):
    real = []
    cache = {}

    for x in relations:
        head_key = tuple(x["head_text"]) if isinstance(x["head_text"], list) else x["head_text"]
        tail_key = tuple(x["tail_text"]) if isinstance(x["tail_text"], list) else x["tail_text"]

        head_name = " ".join(x["head_text"]) if isinstance(x["head_text"], list) else x["head_text"]
        tail_name = " ".join(x["tail_text"]) if isinstance(x["tail_text"], list) else x["tail_text"]

        if head_key not in cache:
            result = get_ticker_safe(head_name)
            cache[head_key] = result[0] if result else None

        if tail_key not in cache:
            result = get_ticker_safe(tail_name)
            cache[tail_key] = result[0] if result else None

        if cache[head_key] is not None and cache[tail_key] is not None:
            real.append(x)
        else:
            rejected = head_name if cache[head_key] is None else tail_name
            print(f"  Rejected: {head_name} -> {tail_name} ('{rejected}' not a known company)")

    return real

def remove_stop_words(text):
    nlp_clean = spacy.blank("en")
    keep = {"from", "for", "to", "by", "with", "of", "or", "and", "such", "as",
            "our", "its", "their", "we", "not", "no", "than", "between"}
    doc = nlp_clean(text)
    tokens = [token.text for token in doc if not token.is_stop or token.lower_ in keep]
    return " ".join(tokens)

def filter_data(relationships, entity):
    filtered = [
        x for x in relationships
        if entity in x["head_text"] or entity in x["tail_text"]
    ]

    seen = set()
    unique = []
    for x in filtered:
        key = (tuple(x["head_text"]), tuple(x["tail_text"]), x["label"])
        if key not in seen:
            seen.add(key)
            unique.append(x)

    best = {}
    for x in unique:
        pair = frozenset([tuple(x["head_text"]), tuple(x["tail_text"])])
        label = x["label"]
        pair_key = (pair, label)
        if pair_key not in best or x["score"] > best[pair_key]["score"]:
            best[pair_key] = x

    ls = list(best.values())

    # Resolve to tickers, remove junk, remove self-referential, deduplicate
    ticker_cache = {}
    real = []
    seen_ticker_pairs = set()

    for x in ls:
        head = " ".join(x["head_text"]) if isinstance(x["head_text"], list) else x["head_text"]
        tail = " ".join(x["tail_text"]) if isinstance(x["tail_text"], list) else x["tail_text"]

        if head not in ticker_cache:
            ticker_cache[head] = get_ticker_safe(head)
        if tail not in ticker_cache:
            ticker_cache[tail] = get_ticker_safe(tail)

        head_ticker = ticker_cache[head]
        tail_ticker = ticker_cache[tail]

        # Both must be real companies
        if not head_ticker or not tail_ticker:
            continue

        # No self-referential
        if head_ticker == tail_ticker:
            continue

        # Deduplicate by ticker pair + label
        ticker_key = (head_ticker, tail_ticker, x["label"])
        if ticker_key in seen_ticker_pairs:
            continue
        seen_ticker_pairs.add(ticker_key)

        real.append({
            "head": head_ticker,
            "head_name": head,
            "tail": tail_ticker,
            "tail_name": tail,
            "label": x["label"],
            "score": x["score"],
        })

    return real


def download_filings(tickers, save_dir="filings"):
    os.makedirs(save_dir, exist_ok=True)
    dl = Downloader("ResearchProject", "research@example.com", save_dir)
    for ticker in tickers:
        # Skip if already downloaded
        if find_filing_file(save_dir, ticker):
            print(f"  {ticker}: already downloaded")
            continue
        print(f"  {ticker}: downloading...")
        try:
            dl.get("10-K", ticker, limit=1, download_details=False)
            time.sleep(0.2)  # Respect SEC rate limits
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")


def find_filing_file(base_dir, ticker):
    """Find the 10-K filing file for a ticker."""
    ticker_dir = os.path.join(base_dir, "sec-edgar-filings", ticker, "10-K")
    if not os.path.exists(ticker_dir):
        return None

    # Prefer HTML files first
    for root, dirs, files in os.walk(ticker_dir):
        for f in files:
            if f.endswith((".html", ".htm")) and "index" not in f.lower():
                return os.path.join(root, f)

    # Then any .txt file, including full-submission
    for root, dirs, files in os.walk(ticker_dir):
        for f in files:
            if f.endswith(".txt"):
                return os.path.join(root, f)

    return None

def extract_10k_from_submission(filepath):
    """Extract just the 10-K document from a full-submission.txt file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split into individual documents
    docs = re.split(r'<DOCUMENT>', content)

    best_doc = None
    best_size = 0

    for doc in docs:
        # Match 10-K but not 10-KSB, exhibits, etc.
        type_match = re.search(r'<TYPE>\s*10-K\s', doc)
        if not type_match:
            # Also try matching the filename pattern
            filename_match = re.search(r'<FILENAME>.*10-k.*\.(htm|html)', doc, re.IGNORECASE)
            if not filename_match:
                continue

        # Extract content between <TEXT> tags
        text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
        raw = text_match.group(1) if text_match else doc

        # Keep the largest matching document (the actual 10-K, not amendments)
        if len(raw) > best_size:
            best_size = len(raw)
            best_doc = raw

    if not best_doc:
        # Fallback: find the largest HTML block in the entire file
        html_blocks = re.findall(r'(<html.*?</html>)', content, re.DOTALL | re.IGNORECASE)
        if html_blocks:
            best_doc = max(html_blocks, key=len)
        else:
            best_doc = content

    # Convert to plaintext
    soup = BeautifulSoup(best_doc, "html.parser")
    for tag in soup(["script", "style", "table", "xbrl",
                     "ix:nonfraction", "ix:nonnumeric",
                     "ix:header", "ix:hidden"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_text_from_filing(filepath):
    """Convert filing to clean plaintext."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "table", "xbrl", "ix:nonfraction",
                     "ix:nonnumeric", "ix:header", "ix:hidden"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # Clean SGML/XBRL artifacts
    text = re.sub(r'<[^>]+>', '', text)  # Any remaining tags
    text = re.sub(r'&\w+;', ' ', text)   # HTML entities
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    SUPPLY_CHAIN_CLUSTER = {
        # Chip designers
        "NVDA": "NVDA",
        "AMD": "AMD",
        "INTC": "Intel",
        # "QCOM": "Qualcomm",
        "AVGO": "Broadcom",
        "TXN": "Texas Instruments",
        "MU": "Micron Technology",
        "MRVL": "Marvell Technology",

        # Semiconductor equipment
        "AMAT": "Applied Materials",
        "LRCX": "Lam Research",
        "KLAC": "KLA Corporation",
        "ON": "ON Semiconductor",
        "ADI": "Analog Devices",
        "MCHP": "Microchip Technology",
        "SWKS": "Skyworks Solutions",


        # Networking / infrastructure
        "CSCO": "Cisco",
        "ANET": "Arista Networks",

        # Contract manufacturing / EMS
        "FLEX": "Flex",
        "JBL": "Jabil",
        "FN": "Fabrinet",
        "CLS": "Celestica",

        # Consumer electronics / big tech
        # "AAPL": "Apple",
        # "MSFT": "Microsoft",
        # "GOOGL": "Alphabet",
        "AMZN": "Amazon",
        "META": "Meta Platforms",
        "TSLA": "Tesla",
        "DELL": "Dell Technologies",
        "HPQ": "HP",
        "HPE": "Hewlett Packard Enterprise",

        # Cloud / enterprise
        "ORCL": "Oracle",
        "IBM": "IBM",
        "CRM": "Salesforce",

        # Automotive / industrial (chip buyers)
        "F": "Ford",
        "GM": "General Motors",
        "DE": "Deere & Company",
        "HON": "Honeywell",
        "GE": "GE Aerospace",
    }

    # download_filings(SUPPLY_CHAIN_CLUSTER.keys())

    for ticker in SUPPLY_CHAIN_CLUSTER.keys():
        file = find_filing_file("filings", ticker)

        text = extract_10k_from_submission(file)

        process_file_stock(text, SUPPLY_CHAIN_CLUSTER[ticker])



def process_file_stock(text, entity):
    nlp = spacy.blank("en")
    nlp.add_pipe("gliner_custom")
    nlp.add_pipe("glirel", after="gliner_custom")

    # text = clean_filing_text(read_text(file_name))


    labels = {"glirel_labels": {
        'manufactures for': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'competitor of': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'purchases from': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'buys from': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'procures from': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'sources from': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'supplies to': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'manufactures for': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'fabricates for': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'assembles for': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'utilizes': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'contracts with': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'is a customer of': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'is a supplier of': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'is a vendor of': {"allowed_head": ["company", "organization"], "allowed_tail": ["company", "organization"]},
        'no relation': {},
    }}

    cleaned = remove_stop_words(resolve_company_references(text, entity))
    chunks = smart_chunk(cleaned)
    chunks = filter_chunks(chunks, nlp)
    print(f"Processing {len(chunks)} chunks...\n")
    all_entities, all_relations = process_all_chunks(chunks, nlp, labels)

    # print("=== All Entities ===")
    # for text, label in set(all_entities):
    #     print(f"  '{text}' -> {label}")
    #
    # all_relations = []
    # for chunk in filter_chunks(chunks, nlp):
    #     docs = list(nlp.pipe([(chunk, labels)], as_tuples=True))
    #     doc = docs[0][0]
    #     # print("=== GLiNER Entities ===")
    #     # for ent in doc.ents:
    #     #     print(f"  '{ent.text}' -> {ent.label_}")
    #     all_relations.extend(doc._.relations)

    relations = filter_data(all_relations, entity)

    print('Number of relations:', len(relations))

    sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
    print("\nDescending Order by Score:")
    for item in sorted_data_desc:
        # if item["score"] > 0.3:
            print(f"{item['head_text']},{item['label']},{item['tail_text']},{item['score']}")

def find_tickers_in_text(text, ticker_set):
    found = set()
    for ticker in ticker_set:
        pattern = re.compile(r'\b' + re.escape(ticker) + r'\b')
        if pattern.search(text):
            found.add(ticker)
    return found

def search_file_for_tickers():
    text = read_text("apple.txt")
    tickers = json.load(open("company_tickers.json"))

    arr = tickers.values()
    tickers = list([x["ticker"] for x in arr])
    titles = list([x["title"] for x in arr])

    for ticker in tickers:
        set = find_tickers_in_text(text, tickers)
        if len(set) > 0:
            print(f"Found ticker: {ticker}")

    for title in titles:
        set = find_tickers_in_text(text, titles)
        if len(set) > 0:
            print(f"Found title: {title}")


if __name__ == '__main__':
    # search_file_for_tickers()
    # start_time = time.time()
    # process_file_stock("qcm.txt", "Qualcomm")
    # end_time = time.time()
    # print("\nTime taken:", end_time - start_time)

    main()

