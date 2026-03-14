import spacy
import glirel


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

    return list(best.values())


def main():
    nlp = spacy.load('en_core_web_trf')

    print("Loading Glirel Data...")
    nlp.add_pipe("glirel", after="ner")
    print("Loading Glirel Data...")

    # text = (
    #     "NVDA have expanded our supplier relationships to build redundancy and resilience "
    #     "in our perations to provide long-term manufacturing capacity aligned with growing "
    #     "customer demand. Our supply chain is mainly concentrated in the Asia-Pacific region. "
    #     "NVDA utilize foundries, such as TSMC, "
    #     "and Samsung, to produce our semiconductor wafers. "
    #     "NVDA purchase memory from SKH, MITH and Samsung. "
    #     "NVDA utilize CoWoS technology for semiconductor packaging. "
    #     "NVDA engage with independent subcontractors and contract manufacturers such as "
    #     "HHPI WSC, and Fabrinet "
    #     "to perform assembly, testing and packaging of our final products."
    # )

    text = """
    Apple purchases MSFT MDM (or thin modem) products, which do not include MSFT integrated application processor technology, and which have lower revenue and margin contributions than our combined modem and application processor products. Consequently, to the extent Apple devices using our MDM products take share from our customers who purchase our integrated modem and application processor products, our revenues and margins may be negatively impacted. Additionally, we expect that Apple will increasingly use its own modem products, rather than our products, in its future devices, which will have a significant negative impact on our QCT revenues, results of operations and cash flows.
    """


    labels = {"glirel_labels": {
        'utilizes' : {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'manufactures for': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'purchases from': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'supplies to': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'assembles for': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'subsidiary of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'acquired by': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'headquartered in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE"]},
        'operates in': {"allowed_head": ["ORG"], "allowed_tail": ["LOC", "GPE"]},
        'partner of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'competitor of': {"allowed_head": ["ORG"], "allowed_tail": ["ORG"]},
        'no relation': {},
    }}

    print("Loading Glirel Data...")
    docs = list( nlp.pipe([(text, labels)], as_tuples=True) )
    # relations = filter_data(docs[0][0]._.relations, "MSFT")
    relations = docs[0][0]._.relations

    print('Number of relations:', len(relations))

    sorted_data_desc = sorted(relations, key=lambda x: x['score'], reverse=True)
    print("\nDescending Order by Score:")
    for item in sorted_data_desc:
        print(f"{item['head_text']} --> {item['label']} --> {item['tail_text']} | score: {item['score']}")

if __name__ == '__main__':
    main()
