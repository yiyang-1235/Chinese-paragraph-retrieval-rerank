"""
metrics.py — 检索评估指标
=========================
实现以下指标：
  Recall@K  ：前 K 个结果中，至少命中一条相关段落的查询比例
  NDCG@K    ：归一化折损累计增益（Normalized DCG），综合考虑命中位置和相关度
  MRR@K     ：平均倒数排名（Mean Reciprocal Rank）

这三个指标与实体链接里的 top-1 准确率是同一类「排序质量」评估，
不同之处在于：
  - 允许多个相关文档（实体链接每条 mention 只有一个正确实体）
  - 看更多候选（@5, @10 而不只 @1）
  - NDCG 还区分相关度高低（score=2 比 score=1 更重要）
"""

import math
from collections import defaultdict


def recall_at_k(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    """
    Recall@K：前 K 个结果中命中至少一条相关段落（score > 0）的查询比例。

    Args:
        run   : {qid: [pid_1, pid_2, ..., pid_n]}  按相关度降序
        qrels : {qid: {pid: relevance_score}}
        k     : 截断位置
    Returns:
        0.0 ~ 1.0
    """
    hits = 0
    total = 0
    for qid, ranked_pids in run.items():
        if qid not in qrels:
            continue
        relevant = {pid for pid, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        total += 1
        top_k = set(ranked_pids[:k])
        if top_k & relevant:        # 交集非空 → 命中
            hits += 1
    return hits / total if total > 0 else 0.0


def ndcg_at_k(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    """
    NDCG@K：归一化折损累计增益。

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
    IDCG@K = 理想排序下的 DCG@K（将相关度从高到低排）
    NDCG@K = DCG@K / IDCG@K

    相关度 rel_i 直接取 qrels 里的 score（0/1 或 0/1/2）。
    """
    ndcg_sum = 0.0
    total = 0
    for qid, ranked_pids in run.items():
        if qid not in qrels:
            continue
        rels = qrels[qid]
        if not any(s > 0 for s in rels.values()):
            continue
        total += 1

        # DCG
        dcg = 0.0
        for rank, pid in enumerate(ranked_pids[:k], start=1):
            rel = rels.get(pid, 0)
            dcg += rel / math.log2(rank + 1)

        # IDCG：把所有相关 score 从高到低排，取前 K 个
        ideal_rels = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))

        ndcg_sum += dcg / idcg if idcg > 0 else 0.0

    return ndcg_sum / total if total > 0 else 0.0


def mrr_at_k(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> float:
    """
    MRR@K：Mean Reciprocal Rank，第一个相关段落排名的倒数均值。
    """
    mrr_sum = 0.0
    total = 0
    for qid, ranked_pids in run.items():
        if qid not in qrels:
            continue
        relevant = {pid for pid, score in qrels[qid].items() if score > 0}
        if not relevant:
            continue
        total += 1
        for rank, pid in enumerate(ranked_pids[:k], start=1):
            if pid in relevant:
                mrr_sum += 1.0 / rank
                break
    return mrr_sum / total if total > 0 else 0.0


def evaluate_all(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """
    一次性计算所有评测指标，返回结果字典。
    """
    return {
        "Recall@1":  recall_at_k(run, qrels, k=1),
        "Recall@5":  recall_at_k(run, qrels, k=5),
        "Recall@10": recall_at_k(run, qrels, k=10),
        "NDCG@10":   ndcg_at_k(run, qrels, k=10),
        "MRR@10":    mrr_at_k(run, qrels, k=10),
    }
