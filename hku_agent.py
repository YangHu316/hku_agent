#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HKU AI对话Agent - 完整版
特点：
1. 支持人设切换
2. 查询扩展
3. 多轮检索
4. 对话记忆
"""

import json
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==================== 配置 ====================
class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_URL = "https://api.deepseek.com/chat/completions" # 注意 URL 修正
    
    # 【修改点】不再使用 API embedding，改用本地轻量级模型
    USE_LOCAL_EMBEDDING = True
    LOCAL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    KNOWLEDGE_BASE_DIR = "knowledge_base"
    TOP_K = 10
    HYBRID_ALPHA = 0.3 # 稍微降低向量权重，防止因模型差异导致检索偏离

# ==================== 简单检索器 ====================
class SimpleRetriever:
    """基于关键词的检索器"""
    
    def __init__(self):
        self.documents = []
    
    def add_document(self, content: str, source: str, metadata: dict = None):
        """添加文档"""
        self.documents.append({
            'content': content,
            'source': source,
            'metadata': metadata or {}
        })
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相关文档"""
        if not self.documents:
            return []
        
        query_keywords = self._extract_keywords(query)
        
        scored_docs = []
        for doc in self.documents:
            score = self._calculate_score(query_keywords, doc['content'])
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]
    
    def _extract_keywords(self, text: str) -> set:
        """提取关键词（中英文）"""
        keywords = set()
        
        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        keywords.update(w for w in english_words if len(w) > 2)
        
        # 中文2-4字词组
        chinese_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        chinese_text = ''.join(chinese_chars)
        
        for length in [2, 3, 4]:
            for i in range(len(chinese_text) - length + 1):
                keywords.add(chinese_text[i:i+length])
        
        # 数字
        numbers = re.findall(r'\d+', text)
        keywords.update(numbers)
        
        return keywords
    
    def _calculate_score(self, query_keywords: set, document: str) -> float:
        """计算文档得分"""
        doc_lower = document.lower()
        doc_keywords = self._extract_keywords(document)
        
        score = 0.0
        
        matched = query_keywords & doc_keywords
        if query_keywords:
            score += len(matched) / len(query_keywords) * 2.0
        
        for keyword in query_keywords:
            if len(keyword) > 1 and keyword in doc_lower:
                score += 0.5
        
        synonyms = {
            'hku': ['香港大学', '港大'],
            '香港大学': ['hku', '港大'],
            '港大': ['hku', '香港大学'],
            '学院': ['faculty', '院系'],
            '排名': ['rank', 'qs', '泰晤士'],
            '成立': ['建立', '创办', '1911'],
        }
        
        for keyword in query_keywords:
            if keyword in synonyms:
                for syn in synonyms[keyword]:
                    if syn in doc_lower:
                        score += 0.3
        
        return score

# ==================== 向量检索器 ====================

class VectorRetriever:
    """基于本地模型的向量检索（稳定、免费）"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def add_document(self, content: str, source: str, metadata: dict = None):
        self.documents.append({
            "content": content,
            "source": source,
            "metadata": metadata or {}
        })

    def build_index(self):
        if not self.documents:
            return
        print("⏳ 正在构建向量索引 (本地模型)...")
        texts = [d["content"] for d in self.documents]
        # encode 直接返回 numpy array
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        print("✅ 索引构建完成")

    def search(self, query: str, top_k: int = 5):
        if self.embeddings is None or not self.documents:
            return []
        
        q_emb = self.model.encode([query], normalize_embeddings=True)
        # 计算相似度
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        
        # 获取 top_k
        idxs = np.argsort(-sims)[:top_k]
        return [self.documents[i] | {"score": float(sims[i])} for i in idxs]


# ==================== Hybrid 检索器 ====================
class HybridRetriever:
    """向量 + 关键词 混合检索"""
    def __init__(self, keyword_ret: SimpleRetriever, vector_ret: VectorRetriever,
                 alpha: float = 0.6):
        self.keyword_ret = keyword_ret
        self.vector_ret = vector_ret
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5,
               keyword_top_k: int = 8, vector_top_k: int = 8):
        kw_docs = self.keyword_ret.search(query, top_k=keyword_top_k)
        vec_docs = self.vector_ret.search(query, top_k=vector_top_k)

        # 用 content 做去重并打分融合
        merged = {}
        for rank, d in enumerate(kw_docs):
            key = d["content"]
            # 关键词得分用 rank 近似：越靠前越高
            kw_score = 1.0 / (rank + 1)
            merged[key] = (d, kw_score, 0.0)

        for rank, d in enumerate(vec_docs):
            key = d["content"]
            vec_score = d.get("score", 1.0 / (rank + 1))
            if key in merged:
                doc, kw_score, _ = merged[key]
                merged[key] = (doc, kw_score, vec_score)
            else:
                merged[key] = (d, 0.0, vec_score)

        scored = []
        for doc, kw_s, vec_s in merged.values():
            final_s = (1 - self.alpha) * kw_s + self.alpha * vec_s
            scored.append((final_s, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ==================== 知识库管理 ====================
class KnowledgeBase:
    """知识库管理器"""
    
    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir

        # 两套检索器
        self.keyword_retriever = SimpleRetriever()

        self.vector_retriever = VectorRetriever(Config.LOCAL_EMBED_MODEL)


        self.retriever = HybridRetriever(
            self.keyword_retriever,
            self.vector_retriever,
            alpha=Config.HYBRID_ALPHA
        )

        os.makedirs(kb_dir, exist_ok=True)


    
    def load(self):
        """加载知识库"""
        print(f"\n📚 加载知识库: {self.kb_dir}")
        
        if not os.listdir(self.kb_dir):
            self._create_samples()
        
        files = [f for f in os.listdir(self.kb_dir) 
                if f.endswith(('.txt', '.md'))]
        
        total_docs = 0
        for filename in files:
            filepath = os.path.join(self.kb_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            paragraphs = [p.strip() for p in content.split('\n\n') 
                         if p.strip() and len(p.strip()) > 2]
            
            for para in paragraphs:
                self.keyword_retriever.add_document(
                    content=para,
                    source=filename,
                    metadata={'length': len(para)}
                )
                self.vector_retriever.add_document(
                    content=para,
                    source=filename,
                    metadata={'length': len(para)}
                )
                total_docs += 1

            
            print(f"  ✓ {filename}: {len(paragraphs)} 段")
        # 建立向量索引
        try:
            self.vector_retriever.build_index()
        except Exception as e:
            print(f"⚠️ 向量索引构建失败，自动降级为关键词检索。错误: {e}")
            self.retriever = self.keyword_retriever


        print(f"✅ 共加载 {total_docs} 个文档片段\n")
    
    def _create_samples(self):
        """创建示例知识库"""
        samples = {
            "hku_basic.txt": """香港大学（The University of Hong Kong，简称HKU或港大）是香港历史最悠久的高等教育机构。

成立时间：1911年3月30日
地理位置：香港岛薄扶林道
校训：明德格物（Sapientia Et Virtus）
学校性质：公立综合性研究型大学""",

            "hku_faculties.txt": """香港大学设有十大学院：

1. 建筑学院 (Faculty of Architecture)
2. 文学院 (Faculty of Arts)
3. 经济及工商管理学院 (Faculty of Business and Economics)
4. 牙医学院 (Faculty of Dentistry)
5. 教育学院 (Faculty of Education)
6. 工程学院 (Faculty of Engineering)
7. 法律学院 (Faculty of Law)
8. 李嘉诚医学院 (Li Ka Shing Faculty of Medicine)
9. 理学院 (Faculty of Science)
10. 社会科学学院 (Faculty of Social Sciences)""",

            "hku_rankings.txt": """香港大学世界排名：

QS 2024：全球第26位
THE 2024：全球第35位

学科优势：
- 牙医学：全球第4位
- 教育学：全球第7位
- 建筑学：全球第14位"""
        }
        
        for filename, content in samples.items():
            filepath = os.path.join(self.kb_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

# ==================== LLM客户端 ====================
class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # 确认 URL 是对的
        self.api_url = "https://api.deepseek.com/chat/completions"
    
    def call(self, messages: List[Dict], temperature: float = 0.3, 
             max_tokens: int = 1000) -> Optional[str]:
        """调用API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            # 【修改点1】把超时时间从 30 改成 60 秒
            # 复杂的人设生成比较慢，30秒经常不够
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                # 【修改点2】打印出具体的错误原因，不要只返回 None
                print(f"\n❌ [API Error] 状态码: {response.status_code}")
                print(f"❌ [API Error] 详情: {response.text}")
                return None
                
        except Exception as e:
            # 【修改点3】打印出网络错误详情
            print(f"\n❌ [Network Error] 连接报错: {e}")
            return None

# ==================== HKU Agent ====================
class HKUAgent:
    """HKU AI对话Agent"""
    
    def __init__(self, kb: KnowledgeBase, llm: LLMClient):
        self.kb = kb
        self.llm = llm
        self.conversation_history = []
        self.persona = None  # 当前人设
    
    def chat(self, user_query: str) -> str:
        """标准对话流程"""
        
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_query}")
        if self.persona:
            print(f"🎭 人设: {self.persona.get('name', '默认')}")
        print(f"{'='*60}\n")
        
        # 1. 查询扩展
        print("🧠 分析查询...")
        expanded_queries = self._expand_query(user_query)
        print(f"✓ 生成 {len(expanded_queries)} 个检索查询\n")
        
        # 2. 多轮检索
        print("🔍 执行检索...")
        all_docs = []
        seen_content = set()
        
        for query in expanded_queries:
            docs = self.kb.retriever.search(query, top_k=3)
            for doc in docs:
                content_hash = hash(doc['content'][:100])
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)
        
        print(f"✓ 找到 {len(all_docs)} 个片段\n")
        
        # 3. 重排序
        if len(all_docs) > Config.TOP_K:
            all_docs = self._rerank_documents(user_query, all_docs)[:Config.TOP_K]
        
        # 4. 生成答案
        print("🤖 AI生成回答...\n")
        answer = self._generate_answer(user_query, all_docs)
        
        # 5. 记录历史
        self.conversation_history.append({
            'user': user_query,
            'assistant': answer,
            'sources': [doc['source'] for doc in all_docs]
        })
        
        return answer
    
    def _expand_query(self, query: str) -> List[str]:
        """查询扩展"""
        
        prompt = f"""你是检索专家。用户问题是："{query}"

生成3-5个不同角度的检索查询，每行一个，不要编号："""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm.call(messages, temperature=0.3, max_tokens=200)
        
        if response:
            queries = [q.strip() for q in response.split('\n') if q.strip()]
            if query not in queries:
                queries.insert(0, query)
            return queries[:5]
        else:
            return [query]
    
    def _rerank_documents(self, query: str, docs: List[Dict]) -> List[Dict]:
        """文档重排序"""
        
        query_keywords = set(self.kb.keyword_retriever._extract_keywords(query))
        
        scored = []
        for doc in docs:
            doc_keywords = set(self.kb.keyword_retriever._extract_keywords(doc['content']))
            overlap = len(query_keywords & doc_keywords)
            relevance = overlap / len(query_keywords) if query_keywords else 0
            scored.append((relevance, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored]
    

    def _generate_answer(self, query: str, docs: List[Dict]) -> str:
        """
        生成答案（极速版：放弃 JSON，使用纯文本解析，大幅提升速度）
        """
        # 1. 兜底
        if not docs:
            return "很抱歉，知识库中暂时没有相关资料。建议直接访问 HKU 官网 (www.hku.hk) 查询。"

        # 2. 构建上下文
        context_blocks = []
        for i, doc in enumerate(docs):
            if len(doc['content']) > 5:
                context_blocks.append(f"[资料{i+1}] {doc['content']}")
        context_str = "\n\n".join(context_blocks)

        # 3. 构建 System Prompt (移除复杂的 JSON 要求)
        base_prompt = self.persona['prompt'] if self.persona else "你是HKU AI助手。"
        
        system_prompt = f"""{base_prompt}

----------------
【回答规则】
1. **事实引用**：涉及客观事实（数据、政策），必须在句尾标注来源，如[1][2]。
2. **人设发挥**：针对用户的情感或咨询，请大胆发挥人设（学长/导师）进行交流，这部分**不需要引用**。
3. **格式要求**：请直接输出回答内容，不要任何 JSON 格式，也不要 Markdown 代码块。
"""

        user_prompt = f"用户问题：{query}\n\n参考资料：\n{context_str}"

        # 4. 调用 LLM (移除 JSON 压力，速度会快很多)
        # max_tokens 限制在 800，防止废话太多导致超时
        raw_response = self.llm.call([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.4, max_tokens=800)

        if not raw_response:
            print("⚠️ [错误] LLM 接口无响应")
            return "抱歉，AI 此时有点繁忙，请再试一次。"

        answer = raw_response.strip()

        # 5. 后端自动提取引用 (替代 JSON 解析)
        # 使用正则从文本中提取 [1], [2] 这样的编号
        import re
        found_citations = re.findall(r'\[(\d+)\]', answer)
        # 去重并转为整数
        citations = sorted(list(set([int(c) for c in found_citations])))
        
        # 默认置信度 (纯文本模式下模型不输出置信度，我们根据是否有引用来给分)
        confidence = 0.85 if citations else 0.5

        # 6. 引用清洗
        valid_ids = set(range(1, len(docs) + 1))
        safe_citations = [c for c in citations if c in valid_ids]
        
        # 7. Verifier 校验 (依然保留，但逻辑不变)
        is_valid = self._verify_answer(query, answer, safe_citations, docs)
        
        if not is_valid:
            print("⚠️ [Verifier] 校验不通过，尝试修正...")
            retry_prompt = f"""你刚才的回答中，客观数据可能有误。
请保持【人设语气】，但修正【客观事实】，确保数据源于以下资料。

资料：
{context_str}

问题：{query}"""
            
            retry_ans = self.llm.call([{"role": "user", "content": retry_prompt}], temperature=0.3)
            if retry_ans:
                answer = retry_ans
                confidence = 0.6
            else:
                return "抱歉，我太想帮你了，但资料不足以支持准确建议。"

        # 8. 最终输出
        cited_sources = sorted(list(set([docs[i-1]['source'] for i in safe_citations])))
        src_text = " | ".join(cited_sources) if cited_sources else "HKU知识库"
        
        return f"{answer}\n\n📊 置信度: {confidence:.2f}\n📚 来源: {src_text}"


    def _verify_answer(self, query: str, answer: str, citations: List[int], docs: List[Dict]) -> bool:
        """
        验证器：严格区分【硬数据】和【软情感】
        """
        # 提取证据
        evidence = []
        for c in citations:
            if 0 < c <= len(docs):
                evidence.append(docs[c-1]['content'])
        evidence_text = "\n".join(evidence)

        # 如果没有引用但回答很长，且包含情感词，放行
        if not evidence_text:
            if len(answer) > 20: 
                return True
            return False

        verifier_prompt = f"""任务：事实核查。

【原则】
1. 只核查【客观事实】（数字、时间、地点、人名、政策）。
2. 忽略【主观内容】（安慰、建议、鼓励、人设语气）。主观内容不需要证据。

问题：{query}
回答：{answer}

证据：
{evidence_text}

请判断：
回答中的【客观事实】是否与证据矛盾？
- 如果只是加了句"别担心"，但数据是对的 -> 输出 YES
- 如果数据错了 -> 输出 NO
- 如果全是安慰话，没提数据 -> 输出 YES

只输出 YES 或 NO。"""

        # 调用 LLM
        res = self.llm.call([{"role": "user", "content": verifier_prompt}], temperature=0.1, max_tokens=5)
        
        # 【关键修复】判空
        if not res:
            # 如果 Verifier 挂了，默认放行（宁可错杀不可不答）
            return True
            
        if "NO" in res.upper():
            return False
        return True


    def _verify_answer(self, query: str, answer: str, citations: List[int], docs: List[Dict]) -> bool:
        """
        验证器：严格区分【硬数据】和【软情感】
        """
        # 提取证据文本
        evidence = []
        for c in citations:
            if 0 < c <= len(docs):
                evidence.append(docs[c-1]['content'])
        evidence_text = "\n".join(evidence)

        # 如果没有引用，但回答很长，可能是纯闲聊，放行
        if not evidence_text and len(answer) > 10:
            return True

        verifier_prompt = f"""任务：事实核查。

【原则】
1. 我们**只核查**客观事实（数字、时间、地点、人名、政策）。
2. 我们**完全忽略**主观内容（安慰、建议、鼓励、人设语气）。主观内容不需要证据。

问题：{query}
回答：{answer}

证据：
{evidence_text}

请判断：
回答中提到的【客观事实】是否与证据矛盾，或凭空捏造了证据中没有的【数据】？
- 如果只是加了句"别担心"，但数据是对的 -> 输出 YES
- 如果数据错了 -> 输出 NO
- 如果全是安慰话，没提数据 -> 输出 YES

只输出 YES 或 NO。"""

        # 温度设为 0.1，让它稍微灵活一点点，别太死板
        res = self.llm.call([{"role": "user", "content": verifier_prompt}], temperature=0.1, max_tokens=5)
        
        if res and "NO" in res.upper():
            return False
        return True


    def _safe_parse_json(self, text: str) -> Optional[dict]:
        """辅助函数：安全的JSON解析"""
        if not text: return None
        try:
            # 1. 尝试直接解析
            return json.loads(text)
        except:
            # 2. 尝试提取 Markdown 代码块 ```json ... ```
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
            # 3. 尝试提取最外层的 {}
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return None


    def _safe_parse_json(self, text: str) -> Optional[dict]:
        """容错解析 JSON"""
        try:
            return json.loads(text)
        except Exception:
            # 尝试截取第一个 {...}
            m = re.search(r'\{.*\}', text, re.S)
            if not m:
                return None
            try:
                return json.loads(m.group())
            except Exception:
                return None
# ==================== 主程序 ====================
def main():
    print("\n🎓 HKU AI对话Agent")
    print("="*60)
    
    kb = KnowledgeBase(Config.KNOWLEDGE_BASE_DIR)
    kb.load()
    
    llm = LLMClient(Config.DEEPSEEK_API_KEY)
    agent = HKUAgent(kb, llm)
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            response = agent.chat(user_input)
            print(f"\n🤖 助手:\n{response}\n")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break

if __name__ == "__main__":
    main()