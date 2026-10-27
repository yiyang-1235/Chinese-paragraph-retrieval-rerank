"""更新简历项目部分为三段式结构"""
import sys, copy
sys.stdout.reconfigure(encoding='utf-8')
from docx import Document
from docx.oxml import OxmlElement

path = r'F:\保研简历\实习简历.docx'
d = Document(path)

def set_text(para, text):
    for run in para.runs:
        run.text = ''
    if para.runs:
        para.runs[0].text = text
    else:
        para.add_run(text)

def add_para_after(para, text):
    new_el = copy.deepcopy(para._element)
    para._element.addnext(new_el)
    new_p = para._element.getnext()
    for child in list(new_p):
        tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
        if tag == 'r':
            new_p.remove(child)
    r = OxmlElement('w:r')
    t = OxmlElement('w:t')
    t.text = text
    r.append(t)
    new_p.append(r)

# P23 标题行不变
# P24 → 技术栈
set_text(d.paragraphs[24],
    '技术栈：Python · jieba · rank_bm25 · BAAI/bge-small-zh-v1.5 · FAISS · BAAI/bge-reranker-base · HuggingFace Datasets')

# P25 → 【项目介绍】
set_text(d.paragraphs[25], '【项目介绍】')

# P26 → 背景
set_text(d.paragraphs[26],
    '· 背景：在 RAG 和搜索类 Agent 中，大模型的回答质量直接依赖于把最相关段落找出来这一步。'
    '传统 BM25 只看词形匹配，遇到语义相关但用词不同的查询会大量漏检；'
    '神经语义检索（双编码器）和交叉编码器重排序能弥补这一缺陷，但速度与精度需要分阶段权衡。')

# P27 → 数据
set_text(d.paragraphs[27],
    '· 数据：使用 T2Ranking（SIGIR 2023，清华大学与搜狗联合发布）作为评测基准，'
    '数据集来源于真实搜索引擎 query，含多级相关性标注（1-3分）；'
    '采用流式下载策略，从 3.5 GB 全量语料中高效提取所需段落，构建本地评测集（500 条查询，50354 条候选段落）。')

# P27 后依次插入方法、成果标题、指标、错误分析
add_para_after(d.paragraphs[27],
    '· 方法：设计三阶段检索对比实验：'
    '① BM25 用 jieba 中文分词建倒排索引；'
    '② BGE 双编码器将查询与段落各自编码为向量，用 FAISS 做近似最近邻检索；'
    '③ BGE-reranker 交叉编码器对双编码器 top-100 候选重新逐对打分精排。')

add_para_after(d.paragraphs[28],
    '【项目成果】')

add_para_after(d.paragraphs[29],
    '· 指标对比（Recall@10 / NDCG@10）：'
    'BM25 83.2%/0.3375 → BGE 94.5%/0.5089 → BGE+重排序 96.0%/0.5591；'
    'BGE 向量检索相比 BM25 基线 NDCG@10 提升 65.7%，重排序在召回率基本不变的前提下进一步改善排名质量（+9.9%）。')

add_para_after(d.paragraphs[30],
    '· 错误分析：BM25 漏检而 BGE 命中的案例主要为语义同义替换'
    '（查询含"领事馆"而段落写"使馆"，词形不重叠但语义相关）；'
    '反向案例（BGE 漏检 BM25 命中）集中于含稀有专有名词的查询，向量模型出现过度泛化。')

d.save(path)
print('完成，输出验证：')
d2 = Document(path)
for i, p in enumerate(d2.paragraphs):
    if 22 <= i <= 34 and p.text.strip():
        print(f'  P{i}: {p.text[:80]}')
