#!/usr/bin/env python3
"""
Generates ui_data.json and a self-contained search_engine.html.

Usage:
    python generate_ui_data.py

After running:
    - Open search_engine.html directly in any browser (self-contained)
    - OR run 'python -m http.server 8000' and open http://localhost:8000/index.html
"""
import json
import os
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(SCRIPT_DIR)

    print("=" * 55)
    print("  BioMed IR Search Engine - Data Generator")
    print("=" * 55)

    print("\n[1/7] Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("[2/7] Loading data files...")
    with open('all_docs_8.jsonl', 'r', encoding='utf-8') as f:
        docs = [json.loads(line) for line in f if line.strip()]
    with open('queries_8.jsonl', 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f if line.strip()]
    qrels = {}
    with open('qrel_8.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                qrels[item['query_id']] = set(item['rel_docs'])
    print(f"   {len(docs)} documents | {len(queries)} queries | {len(qrels)} qrels")

    def clean_text(text):
        doc = nlp(text)
        return " ".join(
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
            and not token.like_url and token.text.strip()
            and token.is_alpha
        )

    doc_ids = [d['doc_id'] for d in docs]
    doc_texts = [f"{d['title']} {d['abstract']}" for d in docs]
    query_texts = [f"{q['title']} {q['need']} {q['context']}" for q in queries]

    print("[3/7] Cleaning text (may take a few minutes)...")
    cleaned_docs = []
    for i, txt in enumerate(doc_texts):
        cleaned_docs.append(clean_text(txt))
        if (i + 1) % 500 == 0:
            print(f"   {i+1}/{len(doc_texts)} documents cleaned...")
    cleaned_queries = [clean_text(txt) for txt in query_texts]
    print(f"   Done: {len(cleaned_docs)} docs, {len(cleaned_queries)} queries")

    print("[4/7] Vectorizing...")
    bin_v = CountVectorizer(binary=True)
    X_bin = bin_v.fit_transform(cleaned_docs)
    Q_bin = bin_v.transform(cleaned_queries)

    freq_v = CountVectorizer()
    X_freq = freq_v.fit_transform(cleaned_docs)
    Q_freq = freq_v.transform(cleaned_queries)

    tfidf_v = TfidfVectorizer()
    X_tfidf = tfidf_v.fit_transform(cleaned_docs)
    Q_tfidf = tfidf_v.transform(cleaned_queries)

    vocab_size = len(bin_v.get_feature_names_out())
    print(f"   Vocabulary: {vocab_size} terms")

    print("[5/7] Computing cosine similarities...")
    reps = {
        'Binary': cosine_similarity(Q_bin, X_bin),
        'Frequency': cosine_similarity(Q_freq, X_freq),
        'TF-IDF': cosine_similarity(Q_tfidf, X_tfidf)
    }

    print("[6/7] Building results and metrics...")
    results = {}
    metrics = {}
    needed_docs = set()

    for rep_name, cos_mat in reps.items():
        results[rep_name] = {}
        metrics[rep_name] = {}
        for q_idx, q in enumerate(queries):
            qid = q['query_id']
            sims = cos_mat[q_idx]
            top_idx = np.argsort(sims)[::-1][:100]

            res_list = []
            for i in top_idx:
                res_list.append({
                    'doc_id': doc_ids[i],
                    'score': round(float(sims[i]), 6)
                })
                needed_docs.add(doc_ids[i])
            results[rep_name][qid] = res_list

            retrieved = {doc_ids[i] for i in top_idx}
            relevant = qrels.get(qid, set())
            tp = len(retrieved & relevant)
            p = tp / len(retrieved) if retrieved else 0
            r = tp / len(relevant) if relevant else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            metrics[rep_name][qid] = {
                'precision': round(p, 4), 'recall': round(r, 4),
                'f1': round(f1, 4), 'tp': tp,
                'relevant_count': len(relevant), 'retrieved_count': 100
            }

    for rd in qrels.values():
        needed_docs.update(rd)

    doc_map = {d['doc_id']: {'title': d['title'], 'abstract': d['abstract']}
               for d in docs if d['doc_id'] in needed_docs}

    ui_data = {
        'queries': queries, 'documents': doc_map,
        'qrels': {k: list(v) for k, v in qrels.items()},
        'results': results, 'metrics': metrics,
        'vocabulary_size': vocab_size, 'num_documents': len(docs)
    }

    print("[7/7] Writing output files...")

    with open('ui_data.json', 'w', encoding='utf-8') as f:
        json.dump(ui_data, f, ensure_ascii=False)
    print(f"   ui_data.json ({len(doc_map)} documents)")

    if os.path.exists('index.html'):
        with open('index.html', 'r', encoding='utf-8') as f:
            html = f.read()
        marker = 'window.UI_DATA = null; // __INJECT_DATA__'
        if marker in html:
            data_str = json.dumps(ui_data, ensure_ascii=False)
            html = html.replace(marker, f'window.UI_DATA = {data_str};')
            with open('search_engine.html', 'w', encoding='utf-8') as f:
                f.write(html)
            print("   search_engine.html (self-contained)")

    print("\n" + "=" * 55)
    print("  Done! Two ways to view:")
    print("  1. Open search_engine.html in your browser")
    print("  2. Run: python -m http.server 8000")
    print("     Then open: http://localhost:8000/index.html")
    print("=" * 55)


if __name__ == '__main__':
    main()
