"""
run_experiment.py — 主实验脚本
================================
依次运行三个版本并输出对比表：

  版本一：BM25（关键词召回）
  版本二：BGE 向量检索（语义召回）
  版本三：BGE 向量检索 + 交叉编码器重排序（精排）

运行前请确保：
  1. pip install -r requirements.txt
  2. python data/prepare_data.py   （下载并预处理数据集，只需运行一次）

运行方式：
  python run_experiment.py

结果会打印到控制台，并保存至 results/results.csv
"""

import csv
import json
import time
from pathlib import Path

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.reranker import Reranker
from src.metrics import evaluate_all

# ── 路径配置 ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
INDEX_DIR   = RESULTS_DIR / "faiss_index"   # FAISS 索引缓存目录

CORPUS_PATH  = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
QRELS_PATH   = DATA_DIR / "qrels.tsv"

# ── 超参数 ────────────────────────────────────────────────────────────────────
BM25_TOP_K       = 100   # BM25 召回数量（也是版本一的最终结果数）
DENSE_TOP_K      = 100   # BGE 召回数量（版本二最终结果数；版本三的输入）
RERANK_TOP_K     = 100   # 交叉编码器重打分的候选数
FINAL_TOP_K      = 100   # 最终返回、计算指标时的截断位置

REBUILD_INDEX    = False  # True：强制重新编码语料库；False：优先复用已保存的索引
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus(path: Path) -> dict[str, str]:
    """返回 {pid: text}"""
    return {row["pid"]: row["text"] for row in load_jsonl(path)}


def load_queries(path: Path) -> dict[str, str]:
    """返回 {qid: text}"""
    return {row["qid"]: row["text"] for row in load_jsonl(path)}


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """返回 {qid: {pid: score}}"""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid   = row["query_id"]
            pid   = row["passage_id"]
            score = int(row["score"])
            qrels.setdefault(qid, {})[pid] = score
    return qrels


def print_results_table(all_results: dict[str, dict[str, float]]) -> None:
    """打印三版对比表。"""
    metrics = ["Recall@1", "Recall@5", "Recall@10", "NDCG@10", "MRR@10"]
    col_w = 14

    header = f"{'方法':<30}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for method, scores in all_results.items():
        row = f"{method:<30}" + "".join(f"{scores[m]:>{col_w}.4f}" for m in metrics)
        print(row)
    print("=" * len(header))


def save_results_csv(
    all_results: dict[str, dict[str, float]],
    path: Path,
) -> None:
    metrics = ["Recall@1", "Recall@5", "Recall@10", "NDCG@10", "MRR@10"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["方法"] + metrics)
        for method, scores in all_results.items():
            writer.writerow([method] + [f"{scores[m]:.4f}" for m in metrics])
    print(f"\n结果已保存至 {path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("── 加载数据 ──")
    corpus  = load_corpus(CORPUS_PATH)
    queries = load_queries(QUERIES_PATH)
    qrels   = load_qrels(QRELS_PATH)
    print(f"  语料库：{len(corpus)} 条段落")
    print(f"  查询数：{len(queries)} 条")
    print(f"  有标注查询数：{len(qrels)} 条\n")

    all_results: dict[str, dict[str, float]] = {}
    timing: dict[str, float] = {}

    # ══════════════════════════════════════════════════════════════════════════
    # 版本一：BM25（关键词召回）
    # ══════════════════════════════════════════════════════════════════════════
    print("── 版本一：BM25 ──")
    t0 = time.time()

    bm25 = BM25Retriever()
    bm25.build(corpus)
    bm25_run = bm25.search(queries, top_k=BM25_TOP_K)

    timing["BM25"] = time.time() - t0
    all_results["版本一：BM25"] = evaluate_all(bm25_run, qrels)
    print(f"  耗时：{timing['BM25']:.1f}s\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 版本二：BGE 向量检索
    # ══════════════════════════════════════════════════════════════════════════
    print("── 版本二：BGE 向量检索 ──")
    t0 = time.time()

    dense = DenseRetriever()
    if not REBUILD_INDEX and (INDEX_DIR / "faiss.index").exists():
        print("  检测到已有 FAISS 索引，直接加载 …")
        dense.load_index(INDEX_DIR)
    else:
        dense.build(corpus)
        dense.save_index(INDEX_DIR)

    dense_run = dense.search(queries, top_k=DENSE_TOP_K)

    timing["BGE"] = time.time() - t0
    all_results["版本二：BGE 向量检索"] = evaluate_all(dense_run, qrels)
    print(f"  耗时：{timing['BGE']:.1f}s\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 版本三：BGE + 交叉编码器重排序
    # ══════════════════════════════════════════════════════════════════════════
    print("── 版本三：BGE + 交叉编码器重排序 ──")
    t0 = time.time()

    reranker = Reranker()
    reranked_run = reranker.rerank(
        first_stage_run=dense_run,
        queries=queries,
        corpus=corpus,
        top_k=FINAL_TOP_K,
        rerank_top=RERANK_TOP_K,
    )

    timing["Rerank"] = time.time() - t0
    all_results["版本三：BGE + 重排序"] = evaluate_all(reranked_run, qrels)
    print(f"  耗时：{timing['Rerank']:.1f}s\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 打印对比表
    # ══════════════════════════════════════════════════════════════════════════
    print_results_table(all_results)
    print(f"\n各版本耗时（检索阶段）：")
    for name, t in timing.items():
        print(f"  {name}: {t:.1f}s")

    save_results_csv(all_results, RESULTS_DIR / "results.csv")

    # ── 错误分析：输出典型失败案例 ────────────────────────────────────────────
    print("\n── 错误分析：BM25 漏检但 BGE 命中的案例（字面匹配弱、语义强）──")
    count = 0
    for qid in list(queries.keys())[:300]:      # 只看前 300 条
        if qid not in qrels:
            continue
        relevant = {pid for pid, s in qrels[qid].items() if s > 0}
        bm25_top5 = set(bm25_run.get(qid, [])[:5])
        dense_top5 = set(dense_run.get(qid, [])[:5])
        if not (bm25_top5 & relevant) and (dense_top5 & relevant):
            print(f"\n  查询：{queries[qid]}")
            hit_pid = next(iter(dense_top5 & relevant))
            print(f"  BGE 命中段落：{corpus.get(hit_pid, '')[:80]}…")
            count += 1
            if count >= 5:
                break

    print("\n── 错误分析：BGE 漏检但 BM25 命中的案例（字面匹配强、语义弱）──")
    count = 0
    for qid in list(queries.keys())[:300]:
        if qid not in qrels:
            continue
        relevant = {pid for pid, s in qrels[qid].items() if s > 0}
        bm25_top5 = set(bm25_run.get(qid, [])[:5])
        dense_top5 = set(dense_run.get(qid, [])[:5])
        if (bm25_top5 & relevant) and not (dense_top5 & relevant):
            print(f"\n  查询：{queries[qid]}")
            hit_pid = next(iter(bm25_top5 & relevant))
            print(f"  BM25 命中段落：{corpus.get(hit_pid, '')[:80]}…")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
