#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HKU Agent 后端服务器 - 支持人设选择
"""

from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import os
# ==================== 【关键修复】 ====================
# 获取 app.py 所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 强制将工作目录切换到 app.py 所在的文件夹
os.chdir(BASE_DIR)
# ====================================================
# 导入统一的hku_agent
from hku_agent import Config, KnowledgeBase, HKUAgent, LLMClient

app = Flask(__name__)
app.secret_key = 'hku-agent-secret-key-2024'
CORS(app, supports_credentials=True)

# 初始化
print("正在初始化Agent...")
kb = KnowledgeBase(Config.KNOWLEDGE_BASE_DIR)
kb.load()
llm = LLMClient(Config.DEEPSEEK_API_KEY)
print("Agent初始化完成！")

# 人设配置库
PERSONA_CONFIGS = {
    "student": {
        "name": "学长/学姐",
        "icon": "🎓",
        "description": "亲身经历HKU学习生活，给你最真实的建议",
        "prompt": """你是HKU的资深学长/学姐，在港大度过了本科和研究生时光。

你最了解：
- 真实的学习体验（课程难度、考试压力、学术氛围）
- 生活的方方面面（住宿、饮食、娱乐、交友）
- 实用的生存技巧（选课攻略、奖学金申请、打工兼职）
- 毕业后的发展（就业市场、深造机会、校友资源）

你的特点：
- 说人话，不打官腔
- 既讲美好，也讲困难
- 给实用建议，不灌鸡汤
- 像朋友聊天，但信息靠谱

回答时：
1. 基于提供的资料，但加入"过来人"的视角
2. 用具体例子说明抽象概念
3. 坦诚优缺点，帮助做决策
4. 提供"如果是我"的建议

---
📊 置信度 | 📚 来源 | 💡 Tips"""
    },
    
    "expert": {
        "name": "全能导师",
        "icon": "👨‍🏫",
        "description": "15年HKU经验，了解学术、生活、发展的方方面面",
        "prompt": """你是一位HKU的全能导师，集多重身份于一身。

知识覆盖：
**学术**：专业设置、课程质量、科研机会
**生活**：住宿、饮食、社交、心理健康
**发展**：实习、就业、深造、校友网络
**实务**：申请、签证、学费、奖学金

回答哲学：
- 📌 真实优先：不美化不丑化
- 🎯 需求导向：理解提问者需求
- 💎 价值增值：提供决策依据
- 🤝 同理关怀：理解焦虑和期待

---
📊 置信度 | 📚 来源 | 💡 建议"""
    },
    
    "researcher": {
        "name": "学术研究者",
        "icon": "🔬",
        "description": "专注高等教育研究，提供深度学术分析",
        "prompt": """你是HKU的资深研究员，专注于高等教育研究。

你的专长：
- 深入研究HKU的学术体系、科研成果
- 了解各学科的发展历史和趋势
- 熟悉HKU在国际学术界的地位
- 掌握详实的数据、排名、论文产出

你的风格：
- 严谨客观，数据说话
- 深入分析，不止表面
- 学术视角，但不艰深
- 批判性思维

---
📊 置信度 | 📚 数据来源 | 🔍 延伸"""
    },
    
    "advisor": {
        "name": "招生顾问",
        "icon": "📋",
        "description": "专业申请指导，帮你成功进入HKU",
        "prompt": """你是香港大学的资深招生顾问，拥有10年招生经验。

你的专长：
- 深入了解HKU的招生政策和录取标准
- 熟悉各专业的申请要求和竞争情况
- 擅长指导文书写作和面试准备
- 了解奖学金评审技巧

---
📊 成功率评估 | 📚 政策依据 | ✅ 行动清单"""
    }
}

# 存储每个session的Agent
agents = {}

def get_agent(session_id: str, persona: str = "expert"):
    """获取或创建Agent"""
    key = f"{session_id}_{persona}"
    
    if key not in agents:
        agent = HKUAgent(kb, llm)
        # 设置人设
        agent.persona = PERSONA_CONFIGS.get(persona, PERSONA_CONFIGS["expert"])
        agents[key] = agent
    
    return agents[key]

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/personas', methods=['GET'])
def get_personas():
    """获取人设列表"""
    personas = []
    for key, config in PERSONA_CONFIGS.items():
        personas.append({
            'id': key,
            'name': config['name'],
            'icon': config['icon'],
            'description': config['description']
        })
    return jsonify({'personas': personas})

@app.route('/set-persona', methods=['POST'])
def set_persona():
    """设置人设"""
    data = request.json
    persona = data.get('persona', 'expert')
    
    session['persona'] = persona
    session['session_id'] = session.get('session_id', os.urandom(16).hex())
    
    return jsonify({'success': True, 'persona': persona})

@app.route('/chat', methods=['POST'])
def chat():
    """对话API"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': '问题不能为空'}), 400
        
        session_id = session.get('session_id', 'default')
        persona = session.get('persona', 'expert')
        
        print(f"\n收到问题: {query}")
        print(f"Session: {session_id}, Persona: {persona}")
        
        # 获取Agent
        agent = get_agent(session_id, persona)
        
        # 调用Agent
        response = agent.chat(query)
        
        # 提取来源
        import re
        sources = re.findall(r'[\w_]+\.txt', response)
        sources = list(set(sources))
        
        return jsonify({
            'answer': response,
            'sources': sources,
            'persona': persona
        })
    
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """重置对话"""
    session_id = session.get('session_id')
    if session_id:
        keys_to_remove = [k for k in agents.keys() if k.startswith(session_id)]
        for key in keys_to_remove:
            del agents[key]
    
    session.clear()
    return jsonify({'success': True})

# ... (上面的代码都不用动) ...

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 HKU Agent 后端服务启动")
    print("="*60)
    print(f"📡 访问: http://localhost:5000")
    print(f"🎭 支持人设切换")
    print("="*60 + "\n")
    
    # ==================== 【新增代码开始】 ====================
    import webbrowser
    from threading import Timer

    def open_browser():
        """延迟1.5秒打开浏览器，给Flask一点启动时间"""
        webbrowser.open('http://localhost:5000')

    # 启动一个定时器，1.5秒后自动打开浏览器
    Timer(1.5, open_browser).start()
    # ==================== 【新增代码结束】 ====================

    # 启动 Flask
    # 注意：use_reloader=False 防止在 debug 模式下浏览器打开两次
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

