"""
bm25_retriever.py — BM25 关键词检索
=====================================
BM25 是信息检索里的经典算法：
  - 不用神经网络，只统计词频（TF）和逆文档频率（IDF）
  - 速度极快，是工业界搜索系统的传统 baseline
  - 缺点：只看词形，「退火车票」≠「退改签规则」，语义匹配弱

实现细节：
  - 用 jieba 对中文分词（段落和查询都要分词）
  - 用 rank_bm25 库的 BM25Okapi 变体（最常用）
  - 参数 k1=1.5, b=0.75（BM25 标准默认值）
"""

import json
from pathlib import Path
from typing import Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


def tokenize_zh(text: str) -> list[str]:
    """
    中文分词：jieba 精确模式，去掉空白 token。
    与实体链接里的分词类似，但这里不去停用词
    （BM25 靠词频区分重要性，去掉停用词反而可能丢失信息）。
    """
    return [tok for tok in jieba.cut(text) if tok.strip()]


class BM25Retriever:
    """
    基于 BM25 的段落检索器。

    使用步骤：
      1. retriever = BM25Retriever()
      2. retriever.build(corpus)   # 建索引（一次）
      3. results = retriever.search(queries, top_k=100)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.bm25: Optional[BM25Okapi] = None
        self.pids: list[str] = []

    def build(self, corpus: dict[str, str]) -> None:
        """
        构建 BM25 倒排索引。
        corpus: {pid: text}
        """
        print("BM25：正在对语料库分词并建索引 …")
        self.pids = list(corpus.keys())
        tokenized = [
            tokenize_zh(corpus[pid])
            for pid in tqdm(self.pids, desc="  分词")
        ]
        # BM25Okapi 直接接受 token 列表的列表
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"  索引完毕，共 {len(self.pids)} 条段落")

    def search(
        self,
        queries: dict[str, str],
        top_k: int = 100,
    ) -> dict[str, list[str]]:
        """
        对所有查询检索，返回 {qid: [pid_1, ..., pid_top_k]}（按分数降序）。
        """
        assert self.bm25 is not None, "请先调用 build() 建索引"
        run: dict[str, list[str]] = {}

        for qid, query_text in tqdm(queries.items(), desc="BM25 检索"):
            q_tokens = tokenize_zh(query_text)
            # get_scores 返回每条段落的 BM25 得分（numpy 数组）
            scores = self.bm25.get_scores(q_tokens)
            # argsort 降序取 top_k
            top_indices = np.argsort(scores)[::-1][:top_k]
            run[qid] = [self.pids[i] for i in top_indices]

        return run
