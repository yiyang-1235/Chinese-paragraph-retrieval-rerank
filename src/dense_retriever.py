"""
dense_retriever.py — BGE 向量检索（双编码器）
=============================================
BGE（BAAI General Embedding）是北京智源研究院开源的中文向量模型，
在 C-MTEB 检索榜单上效果最好的小模型之一。

双编码器（Bi-encoder）工作方式：
  ┌──────────┐         ┌──────────┐
  │  查询文本 │─→ BGE ─→│ 查询向量 │──┐
  └──────────┘         └──────────┘  │  余弦相似度 → 排序
  ┌──────────┐         ┌──────────┐  │
  │  段落文本 │─→ BGE ─→│ 段落向量 │──┘
  └──────────┘         └──────────┘

与实体链接的 HAN 的关系：
  - 两者都把文本编成定长向量，再算余弦相似度
  - HAN 是自己训练的；BGE 是别人在大规模中文文本对上训练好的
  - 段落向量可以离线预先算好（encode_corpus），查询来了再实时编码

速度权衡：
  - encode_corpus：段落库编码一次，之后可以重用
  - 向量检索用 FAISS：从 2 万段落中找 top-100，毫秒级
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# BGE-small-zh：~100MB，RTX 4050 上单批次约 200ms
BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# BGE 官方推荐：检索时在查询前加指令，段落不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class DenseRetriever:
    """
    基于 BGE 向量模型 + FAISS 的稠密段落检索器。

    使用步骤：
      1. retriever = DenseRetriever()
      2. retriever.build(corpus)          # 编码段落 + 建 FAISS 索引（耗时较长）
      3. results = retriever.search(queries, top_k=100)

    build() 结果可以持久化（save_index / load_index），
    避免每次实验重新编码语料库。
    """

    def __init__(self, model_name: str = BGE_MODEL_NAME, batch_size: int = 256):
        print(f"加载向量模型：{model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size  = batch_size
        self.index: Optional[faiss.IndexFlatIP] = None   # 内积 = 余弦（向量已归一化）
        self.pids: list[str] = []

    # ── 编码 ──────────────────────────────────────────────────────────────────

    def _encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """
        批量编码文本，返回 L2 归一化后的向量矩阵（float32）。
        BGE 要求：向量归一化后才能用内积代替余弦相似度。
        """
        if prefix:
            texts = [prefix + t for t in texts]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2 归一化，和实体链接里的 F.normalize 效果一样
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # ── 建索引 ─────────────────────────────────────────────────────────────────

    def build(self, corpus: dict[str, str]) -> None:
        """
        编码所有段落并建 FAISS 平坦内积索引（IndexFlatIP）。
        IndexFlatIP：暴力搜索，结果精确，对 <10 万段落完全够用。
        """
        self.pids = list(corpus.keys())
        texts = [corpus[pid] for pid in self.pids]
        dim   = self.model.get_sentence_embedding_dimension()

        print("BGE：正在编码语料库 …")
        embeddings = self._encode(texts)   # [N, dim]

        print("构建 FAISS 索引 …")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"  FAISS 索引完毕，共 {self.index.ntotal} 条向量")

    # ── 持久化 ─────────────────────────────────────────────────────────────────

    def save_index(self, save_dir: Path) -> None:
        """把 FAISS 索引和段落 ID 列表存到磁盘，方便复用。"""
        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "faiss.index"))
        (save_dir / "pids.json").write_text(
            json.dumps(self.pids, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  索引已保存至 {save_dir}")

    def load_index(self, save_dir: Path) -> None:
        """从磁盘加载已建好的 FAISS 索引。"""
        self.index = faiss.read_index(str(save_dir / "faiss.index"))
        self.pids  = json.loads((save_dir / "pids.json").read_text(encoding="utf-8"))
        print(f"  索引已加载，共 {self.index.ntotal} 条向量")

    # ── 检索 ──────────────────────────────────────────────────────────────────

    def search(
        self,
        queries: dict[str, str],
        top_k: int = 100,
    ) -> dict[str, list[str]]:
        """
        对所有查询做向量检索，返回 {qid: [pid_1, ..., pid_top_k]}（按相似度降序）。
        查询文本前加 BGE 官方检索指令，段落不加。
        """
        assert self.index is not None, "请先调用 build() 或 load_index()"

        qids  = list(queries.keys())
        texts = [queries[qid] for qid in qids]

        print("BGE：正在编码查询 …")
        q_embeddings = self._encode(texts, prefix=QUERY_INSTRUCTION)  # [Q, dim]

        print("FAISS 向量检索 …")
        # search 返回 (scores, indices)，scores 是内积，indices 是在 self.pids 中的下标
        _, all_indices = self.index.search(q_embeddings, top_k)       # [Q, top_k]

        run: dict[str, list[str]] = {}
        for i, qid in enumerate(qids):
            run[qid] = [
                self.pids[idx]
                for idx in all_indices[i]
                if idx != -1   # FAISS 返回 -1 表示没有足够结果
            ]
        return run
