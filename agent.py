import os
import re
import calendar
from datetime import date, timedelta
from typing import List, Optional, Tuple, Iterable, Dict, Any
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

# =========================
# 配置与常量
# =========================

FIXED_HOLIDAYS = {
    "元旦": (1, 1),
    "劳动节": (5, 1),
    "儿童节": (6, 1),
    "教师节": (9, 10),
    "国庆节": (10, 1),
}

@dataclass
class AgentConfig:
    kb_path: str = "data/knowledge_base.md"
    embed_model: str = "shibing624/text2vec-base-chinese"
    top_k: int = 6
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

# =========================
# 日期工具类 (Deterministic Calculation)
# =========================

class DateUtils:
    @staticmethod
    def is_leap_year(y: int) -> bool:
        return (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)

    @staticmethod
    def days_in_month(y: int, m: int) -> int:
        return calendar.monthrange(y, m)[1]

    @staticmethod
    def month_size_label(m: int) -> str:
        if m == 2: return "特殊"
        if m in (1, 3, 5, 7, 8, 10, 12): return "大月"
        if m in (4, 6, 9, 11): return "小月"
        return "未知"

    @staticmethod
    def parse_ymd(text: str) -> Optional[Tuple[int, int, int]]:
        patterns = [
            r"(?P<y>\d{4})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日?",
            r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})",
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return int(match.group("y")), int(match.group("m")), int(match.group("d"))
        return None

    @staticmethod
    def parse_date(text: str) -> Optional[date]:
        res = DateUtils.parse_ymd(text)
        if not res: return None
        try:
            return date(*res)
        except ValueError:
            return None

    @staticmethod
    def extract_year(text: str) -> Optional[int]:
        m = re.search(r"(?P<y>\d{4})\s*年", text)
        if m: return int(m.group("y"))
        m2 = re.search(r"\b(?P<y>\d{4})\b", text)
        if m2: return int(m2.group("y"))
        return None

    @staticmethod
    def extract_n_days(text: str) -> Optional[int]:
        m = re.search(r"(?P<n>\d+)\s*(?:天|日)", text)
        if m: return int(m.group("n"))
        return None

    @staticmethod
    def get_quarter_days(y: int, q: int) -> int:
        if q == 1: return 31 + (29 if DateUtils.is_leap_year(y) else 28) + 31
        if q == 2: return 30 + 31 + 30
        if q == 3: return 31 + 31 + 30
        if q == 4: return 31 + 30 + 31
        return 0

# =========================
# Agent 核心逻辑
# =========================

class DateAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._vs = None
        self._client = None

    def _get_client(self):
        if not self._client and self.config.api_key:
            # 增加超时时间到 60 秒
            self._client = OpenAI(
                api_key=self.config.api_key, 
                base_url=self.config.base_url,
                timeout=60.0
            )
        return self._client

    def _load_vectorstore(self):
        if self._vs: return self._vs
        
        if not os.path.exists(self.config.kb_path):
            raise FileNotFoundError(f"Knowledge base not found: {self.config.kb_path}")

        with open(self.config.kb_path, "r", encoding="utf-8") as f:
            kb_text = f.read()

        # 按 CARD 切分
        parts = []
        buf = []
        for line in kb_text.splitlines():
            if line.startswith("## CARD-") and buf:
                parts.append("\n".join(buf).strip())
                buf = [line]
            else:
                buf.append(line)
        if buf: parts.append("\n".join(buf).strip())

        docs = [Document(page_content=p, metadata={"source": "kb"}) for p in parts if p]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n---\n", "\n## ", "\n- ", "\n", "。", "；", " ", ""],
        )
        split_docs = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embed_model)
        self._vs = FAISS.from_documents(split_docs, embeddings)
        return self._vs

    def detect_and_compute(self, text: str) -> Dict[str, Any]:
        """确定性计算逻辑"""
        year = DateUtils.extract_year(text)
        dt = DateUtils.parse_date(text)
        
        # 1. 闰年判断
        if any(k in text for k in ["闰年", "平年", "是不是闰年", "是不是平年", "闰不闰"]):
            if year is None:
                return {"ok": False, "intent": "leap_year", "error": "请提供年份，例如“2026年”"}
            return {"ok": True, "intent": "leap_year", "year": year, "is_leap": DateUtils.is_leap_year(year)}

        # 2. 季度天数
        q_match = re.search(r"第(?P<q>[1234一二三四])季度", text)
        if q_match:
            q_map = {"1":1, "2":2, "3":3, "4":4, "一":1, "二":2, "三":3, "四":4}
            q_val = q_map.get(q_match.group("q"))
            y_val = year or date.today().year
            return {"ok": True, "intent": "quarter_days", "year": y_val, "quarter": q_val, "days": DateUtils.get_quarter_days(y_val, q_val)}

        # 3. 某月天数/类型
        m_match = re.search(r"(?P<m>\d{1,2})\s*月", text)
        if m_match and any(k in text for k in ["多少天", "几天", "大月", "小月", "什么月"]):
            m_val = int(m_match.group("m"))
            if 1 <= m_val <= 12:
                y_val = year or date.today().year
                kind = DateUtils.month_size_label(m_val)
                days = DateUtils.days_in_month(y_val, m_val)
                return {"ok": True, "intent": "month_info", "year": y_val, "month": m_val, "kind": kind, "days": days}

        # 4. 星期计算
        if any(k in text for k in ["星期几", "周几", "星期", "周"]):
            if dt:
                cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
                return {"ok": True, "intent": "weekday", "date": dt.isoformat(), "weekday": cn[dt.weekday()]}
            elif DateUtils.parse_ymd(text):
                 return {"ok": False, "intent": "weekday", "error": "日期不合法（如2月30日）"}

        # 5. 日期偏移 (往后/往前 N 天)
        if dt and any(k in text for k in ["后", "前", "往后", "往前"]):
            n = DateUtils.extract_n_days(text)
            if n is not None:
                is_forward = any(k in text for k in ["后", "往后"])
                delta = n if is_forward else -n
                # 默认口径：如果是“持续N天”，通常包含当天。但这里是“N天后”，通常不含当天。
                # 按照知识库 CARD-110，我们将根据关键词微调。
                if "持续" in text:
                    res_dt = dt + timedelta(days=n-1 if is_forward else -n+1)
                else:
                    res_dt = dt + timedelta(days=delta)
                return {"ok": True, "intent": "offset_days", "base": dt.isoformat(), "days": n, "direction": "后" if is_forward else "前", "result": res_dt.isoformat()}

        # 6. 节日倒计时
        if any(h in text for h in FIXED_HOLIDAYS) and any(k in text for k in ["还有几天", "多少天", "距离"]):
            holiday = next(h for h in FIXED_HOLIDAYS if h in text)
            start = dt or date.today()
            h_m, h_d = FIXED_HOLIDAYS[holiday]
            target = date(start.year, h_m, h_d)
            if target < start: target = date(start.year + 1, h_m, h_d)
            return {"ok": True, "intent": "holiday_countdown", "holiday": holiday, "start": start.isoformat(), "target": target.isoformat(), "days": (target - start).days}

        # 7. 日期存在性
        if any(k in text for k in ["存在", "合法", "有没有"]):
            ymd = DateUtils.parse_ymd(text)
            if ymd:
                try:
                    date(*ymd)
                    return {"ok": True, "intent": "validity", "valid": True, "date": f"{ymd[0]}-{ymd[1]}-{ymd[2]}"}
                except ValueError:
                    return {"ok": True, "intent": "validity", "valid": False, "date": f"{ymd[0]}-{ymd[1]}-{ymd[2]}"}

        return {"ok": False, "intent": "unknown"}

    def ask(self, query: str, role: str = "小学生") -> Iterable[str]:
        # 获取当前真实日期
        today = date.today()
        current_date_str = today.strftime("%Y年%m月%d日")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][today.weekday()]

        # 1. 尝试确定性计算
        payload = self.detect_and_compute(query)
        
        # 2. 检索知识库
        try:
            vs = self._load_vectorstore()
            docs = vs.similarity_search(query, k=self.config.top_k)
            context = "\n\n".join([f"[片段{i+1}]\n{d.page_content}" for i, d in enumerate(docs)])
        except Exception as e:
            context = f"（检索失败: {e}）"
            docs = []

        # 3. 构造 Prompt 并调用 LLM
        client = self._get_client()
        if not client:
            # 离线兜底
            yield self._offline_fallback(query, payload, docs)
            return

        system_prompt = self._build_system_prompt(role)
        prompt_content = (
            f"【当前现实时间】\n今天是：{current_date_str}，{current_weekday_cn}\n\n"
            f"【任务说明】\n你是一位专业的日期知识专家。请根据提供的知识库和计算结果回答用户问题。\n"
            f"【重要要求】\n回答中禁止提及“CARD-xxx”、“知识卡片”或任何内部编号。请直接以老师的身份输出自然、专业的回答内容。\n"
            f"【知识库参考】\n{context}\n\n"
            f"【计算工具结果】\n{payload}\n\n"
            f"【输出规范】\n"
            f"1. 结论优先：直接给出准确答案。\n"
            f"2. 步骤清晰：简单解释推导过程。\n"
            f"3. 规则引用：引用知识库中的规则（如“四年一闰”）。\n"
            f"4. 口径说明：涉及天数计算时说明是否包含当天。\n\n"
            f"【用户提问】\n{query}"
        )

        try:
            stream = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content},
                ],
                temperature=0.3,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"（LLM 调用出错: {e}，改用离线模式）\n\n"
            yield self._offline_fallback(query, payload, docs)

    def _build_system_prompt(self, role: str) -> str:
        base_instruction = "\n注意：绝对不要在回答中提及“CARD-xxx”、“知识卡片”或任何内部编号。"
        if role == "家长":
            return "你是一位严谨的数学教育专家，面向家长提供准确的日期知识解答，强调规则、口径和易错点。" + base_instruction
        return "你是一位温柔耐心的日期小老师，用简单有趣的语言给小学生讲解日期知识，步骤分明，多用鼓励性语言。" + base_instruction

    def _offline_fallback(self, query: str, payload: dict, docs: List[Document]) -> str:
        if payload.get("ok"):
            intent = payload["intent"]
            if intent == "leap_year":
                return f"【离线回答】{payload['year']}年{'是' if payload['is_leap'] else '不是'}闰年。\n\n**依据**：根据公历规则，四年一闰，百年不闰，四百年再闰。"
            if intent == "weekday":
                return f"【离线回答】{payload['date']} 是 **{payload['weekday']}**。"
            if intent == "month_info":
                return f"【离线回答】{payload['year']}年{payload['month']}月是 **{payload['kind']}**，共有 {payload['days']} 天。"
            if intent == "holiday_countdown":
                return f"【离线回答】距离 {payload['holiday']} 还有 **{payload['days']}** 天。"
            if intent == "offset_days":
                return f"【离线回答】从 {payload['base']} 往{payload['direction']} {payload['days']} 天的结果是：**{payload['result']}**。"
        
        if docs:
            # 优化展示：彻底清理内部编号和元数据标签
            content = docs[0].page_content
            
            # 移除 CARD 编号行
            content = re.sub(r"## CARD-\d+.*?\n", "", content)
            
            # 如果包含 Content: 标签，则只取 Content 部分
            if "Content:" in content:
                # 尝试提取 Content 到 Output 之间的核心内容
                match = re.search(r"Content:(.*?)(?:Output:|$)", content, re.DOTALL)
                if match:
                    cleaned_content = match.group(1).strip()
                else:
                    cleaned_content = content.split("Content:")[1].strip()
            else:
                # 如果没有 Content 标签，则过滤掉常见的元数据行
                lines = content.splitlines()
                filtered_lines = [
                    line for line in lines 
                    if not any(line.strip().startswith(prefix) for prefix in [
                        "- Title:", "- Alias:", "- Tags:", "- Trigger:", "- Output:", "## CARD-"
                    ])
                ]
                cleaned_content = "\n".join(filtered_lines).strip()
            
            # 移除多余的空行
            cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)
            
            return f"【离线提示】我暂时无法连接云端大脑，不过我在知识库里为你找到了相关规则：\n\n{cleaned_content}\n\n（您可以尝试刷新页面，恢复云端智能对话体验）"
        
        return "抱歉，由于网络波动，我目前无法连接云端大脑。请检查您的 API 配置或网络连接，稍后再试。"

