"""
Retrieval + answer evaluation harness.

Usage:
    python eval/run_eval.py eval_set.json
    python eval/run_eval.py eval_set.json --out results.json
    python eval/run_eval.py eval_set.json --full          # also score answers
    python eval/run_eval.py eval_set.json --ids dom-05 for-12  # subset

Metrics:
    context_recall    -- fraction of ground-truth contexts covered by retrieved chunks
    context_precision -- fraction of retrieved chunks that cover a GT context
    answer_correct    -- (--full only) 1 if any GT keyword string appears in the answer

A retrieved chunk "covers" a GT context when their word-level Jaccard similarity
exceeds OVERLAP_THRESHOLD (default 0.25).  Tune if your passages are very short.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Allow running from the project root or from eval/.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

from config.settings import settings
from pipelines.rag_pipeline import RAGPipeline
from retrieval.decomposer import decompose_query
from retrieval.hyde import generate_hyde
from retrieval.multi_query import generate_query_variants
from retrieval.reranker import rerank_and_filter
from retrieval.retrievers import dedupe_docs
from retrieval.rewrite import rewrite_query

OVERLAP_THRESHOLD = 0.25  # word-Jaccard: chunk vs GT context


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _word_jaccard(a: str, b: str) -> float:
    a_w = set(a.lower().split())
    b_w = set(b.lower().split())
    union = a_w | b_w
    return len(a_w & b_w) / len(union) if union else 0.0


def _chunk_covers_context(chunk_text: str, gt_context: str) -> bool:
    return _word_jaccard(chunk_text, gt_context) >= OVERLAP_THRESHOLD


# ---------------------------------------------------------------------------
# Per-item retrieval (mirrors rag_pipeline.stream_answer without the LLM call)
# ---------------------------------------------------------------------------

def _run_retrieval(pipeline: RAGPipeline, question: str) -> List[Document]:
    rewritten = rewrite_query(question, [])
    sources = pipeline.router.route(rewritten)

    sub_queries = (
        decompose_query(rewritten) if settings.decomposer_enabled else [rewritten]
    )

    candidates: List[Document] = []
    for i, sq in enumerate(sub_queries):
        vec_query = generate_hyde(sq) if (settings.hyde_enabled and i == 0) else sq
        candidates += pipeline._retrieve_candidates(sq, vec_query, sources)

    if pipeline._should_do_global_pass(sources):
        for i, sq in enumerate(sub_queries):
            vec_query = generate_hyde(sq) if (settings.hyde_enabled and i == 0) else sq
            candidates += pipeline._retrieve_candidates(sq, vec_query, None)

    candidates = dedupe_docs(candidates)

    if settings.multi_query_enabled:
        for variant in generate_query_variants(rewritten)[1:]:
            for doc in pipeline.bm25_by_source["all"].invoke(variant):
                candidates.append(doc)
        candidates = dedupe_docs(candidates)

    kept = rerank_and_filter(rewritten, candidates, pipeline.cross_encoder)
    return kept, rewritten


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_item(pipeline: RAGPipeline, item: dict, full: bool) -> dict:
    question = item["question"]
    gt_contexts: List[str] = item["contexts"]

    kept, rewritten = _run_retrieval(pipeline, question)

    # Context recall: what fraction of GT contexts are covered?
    recall_hits = [
        any(_chunk_covers_context(doc.page_content, gt) for doc in kept)
        for gt in gt_contexts
    ]
    recall = sum(recall_hits) / len(recall_hits) if recall_hits else 0.0

    # Context precision: what fraction of retrieved chunks are relevant?
    precision_hits = [
        any(_chunk_covers_context(doc.page_content, gt) for gt in gt_contexts)
        for doc in kept
    ]
    precision = sum(precision_hits) / len(precision_hits) if precision_hits else 0.0

    result = {
        "id": item["id"],
        "question": question,
        "policy": item.get("policy"),
        "difficulty": item.get("difficulty"),
        "type": item.get("type"),
        "routed_to": pipeline.router.route(rewritten),
        "num_candidates_after_rerank": len(kept),
        "context_recall": round(recall, 4),
        "context_precision": round(precision, 4),
        "recall_per_context": recall_hits,
        "missed_contexts": [
            gt for gt, hit in zip(gt_contexts, recall_hits) if not hit
        ],
    }

    if full:
        # Full answer — collect streamed tokens.
        answer_tokens = list(pipeline.stream_answer(question))
        answer = "".join(answer_tokens)
        pipeline.memory.clear()  # don't bleed state between eval items

        # Rough correctness: check if the ground-truth key phrases appear.
        gt_answer = item.get("ground_truth", "")
        # Split GT into ~5-word windows and check coverage.
        gt_words = gt_answer.lower().split()
        windows = [
            " ".join(gt_words[i : i + 5]) for i in range(0, max(1, len(gt_words) - 4))
        ]
        covered = sum(1 for w in windows if w in answer.lower())
        answer_score = covered / len(windows) if windows else 0.0

        result["answer"] = answer
        result["answer_keyword_score"] = round(answer_score, 4)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(results: list, full: bool) -> None:
    n = len(results)
    avg_recall = sum(r["context_recall"] for r in results) / n
    avg_prec = sum(r["context_precision"] for r in results) / n

    print(f"\n{'='*60}")
    print(f"OVERALL   n={n}   recall={avg_recall:.3f}   precision={avg_prec:.3f}")
    if full:
        avg_ans = sum(r.get("answer_keyword_score", 0) for r in results) / n
        print(f"          answer_keyword_score={avg_ans:.3f}")

    # Break down by policy.
    for policy in sorted({r.get("policy") for r in results if r.get("policy")}):
        sub = [r for r in results if r.get("policy") == policy]
        sr = sum(r["context_recall"] for r in sub) / len(sub)
        sp = sum(r["context_precision"] for r in sub) / len(sub)
        print(f"  {policy}   n={len(sub)}   recall={sr:.3f}   precision={sp:.3f}")

    # Break down by difficulty.
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in results if r.get("difficulty") == diff]
        if sub:
            sr = sum(r["context_recall"] for r in sub) / len(sub)
            sp = sum(r["context_precision"] for r in sub) / len(sub)
            print(f"  {diff:<6}   n={len(sub)}   recall={sr:.3f}   precision={sp:.3f}")

    # Failures.
    failures = [r for r in results if r["context_recall"] < 1.0]
    if failures:
        print(f"\nFAILED RETRIEVAL ({len(failures)} items, recall < 1.0):")
        for r in failures:
            print(f"  [{r['id']}] recall={r['context_recall']:.2f}  "
                  f"{r['question'][:65]}...")
            for ctx in r["missed_contexts"]:
                print(f"         missed: {ctx[:80]}...")
    else:
        print("\nAll items retrieved with recall = 1.0")
    print("="*60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval/answer eval harness")
    parser.add_argument("eval_file", help="Path to eval set JSON")
    parser.add_argument(
        "--full", action="store_true",
        help="Also generate answers and compute keyword-overlap answer score"
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Write per-item results to a JSON file"
    )
    parser.add_argument(
        "--ids", nargs="+", metavar="ID",
        help="Evaluate only these item IDs (e.g. --ids dom-05 for-12)"
    )
    args = parser.parse_args()

    with open(args.eval_file, encoding="utf-8") as f:
        data = json.load(f)
    items = data["data"] if "data" in data else data

    if args.ids:
        items = [it for it in items if it["id"] in args.ids]
        if not items:
            sys.exit(f"No items matched IDs: {args.ids}")

    print(f"Loading pipeline... (chunk_size={settings.chunk_size}, "
          f"vector_k={settings.vector_k}, bm25_k={settings.bm25_k}, "
          f"rerank_top_n={settings.rerank_top_n}, "
          f"threshold={settings.rerank_score_threshold})")
    pipeline = RAGPipeline()

    results = []
    for item in items:
        print(f"\n[{item['id']}] {item['question'][:70]}...")
        result = _score_item(pipeline, item, full=args.full)
        results.append(result)
        print(f"  recall={result['context_recall']:.2f}  "
              f"precision={result['context_precision']:.2f}  "
              f"kept={result['num_candidates_after_rerank']}")
        if args.full and "answer_keyword_score" in result:
            print(f"  answer_score={result['answer_keyword_score']:.2f}")

    _print_summary(results, full=args.full)

    if args.out:
        payload = {
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "vector_k": settings.vector_k,
                "bm25_k": settings.bm25_k,
                "rerank_top_n": settings.rerank_top_n,
                "rerank_score_threshold": settings.rerank_score_threshold,
                "decomposer_enabled": settings.decomposer_enabled,
                "routing_strict_filter": settings.routing_strict_filter,
            },
            "summary": {
                "n": len(results),
                "avg_context_recall": round(
                    sum(r["context_recall"] for r in results) / len(results), 4
                ),
                "avg_context_precision": round(
                    sum(r["context_precision"] for r in results) / len(results), 4
                ),
            },
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
