"""
reranker.py — BGE 交叉编码器重排序
=====================================
交叉编码器（Cross-encoder）与双编码器的核心区别：

  双编码器（BGE 召回）：
    查询 → 向量A    ↘
                     点积 → 分数
    段落 → 向量B    ↗
    ✓ 段落向量可离线预算，查询实时编码，速度快
    ✗ 两段文本没有跨词交互，准确性有上限

  交叉编码器（BGE 重排序）：
    [查询 + 段落] → 模型 → 分数
    ✓ 能看到查询词和段落词之间的细粒度匹配（attention 全互连）
    ✗ 必须逐对计算，不能离线缓存；只能用在少量候选上

实际系统的做法（这个项目复现的流程）：
  阶段一（召回）：BM25 或 BGE 从全库快速找出 top-100
  阶段二（精排）：交叉编码器只对这 100 条重新打分、排序
  → 速度和精度兼得

与实体链接项目的对应：
  你的 EntityLinker.scorer（MLP）就是一个轻量级的「重排序器」，
  接收 [Sim1, Sim2] 然后输出得分。
  交叉编码器做的是同一件事，只是输入从两个相似度升级成了拼接文本对，
  参数量从 2→64→1 升级成了 1 亿参数的 BERT 变体。
"""

from typing import Optional

from sentence_transformers import CrossEncoder
from tqdm import tqdm

BGE_RERANKER_NAME = "BAAI/bge-reranker-base"   # ~280MB，RTX 4050 完全支持


class Reranker:
    """
    BGE 交叉编码器重排序器。

    使用步骤：
      1. reranker = Reranker()
      2. reranked_run = reranker.rerank(
             first_stage_run,    # 一阶段检索结果 {qid: [pid, ...]}
             queries,            # {qid: text}
             corpus,             # {pid: text}
             top_k=10,           # 最终保留的数量
             rerank_top=100,     # 对前多少个重打分
         )
    """

    def __init__(self, model_name: str = BGE_RERANKER_NAME, batch_size: int = 64):
        print(f"加载重排序模型：{model_name}")
        # CrossEncoder 直接接受 (query, passage) 对，不需要分别编码
        self.model      = CrossEncoder(model_name, max_length=512)
        self.batch_size = batch_size

    def rerank(
        self,
        first_stage_run: dict[str, list[str]],
        queries: dict[str, str],
        corpus: dict[str, str],
        top_k: int = 100,
        rerank_top: int = 100,
    ) -> dict[str, list[str]]:
        """
        对一阶段召回结果重新打分并排序。

        Args:
            first_stage_run : {qid: [pid, ...]}  一阶段检索结果
            queries         : {qid: text}
            corpus          : {pid: text}
            top_k           : 最终返回的段落数
            rerank_top      : 对前多少个进行重排（rerank_top ≤ len(first_stage) ）
        Returns:
            {qid: [pid, ...]}  重排后的结果（取前 top_k 个）
        """
        reranked_run: dict[str, list[str]] = {}

        for qid, candidates in tqdm(first_stage_run.items(), desc="交叉编码器重排序"):
            query_text = queries.get(qid, "")
            # 只对前 rerank_top 个重打分（其余直接截断）
            cands_to_rerank = candidates[:rerank_top]

            # 构造 (query, passage) 对
            pairs = [
                (query_text, corpus.get(pid, ""))
                for pid in cands_to_rerank
            ]

            if not pairs:
                reranked_run[qid] = []
                continue

            # predict 返回每对的相关性分数（标量）
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # 按分数降序排列
            sorted_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )
            reranked_run[qid] = [cands_to_rerank[i] for i in sorted_indices[:top_k]]

        return reranked_run
