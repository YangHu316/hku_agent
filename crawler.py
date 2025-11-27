#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能知识收集器
策略：
1. 优先用API获取框架
2. 提示用户补充细节
3. 自动整理保存
"""

import requests
import os
import time
import json

class SmartCollector:
    """智能收集器"""
    
    def __init__(self):
        self.kb_dir = "knowledge_base"
        os.makedirs(self.kb_dir, exist_ok=True)
    
    def fetch_api_summary(self, keyword):
        """从API获取摘要"""
        print(f"\n📡 正在从API获取 {keyword} 的基本信息...")
        
        api_url = "https://baike.baidu.com/api/openapi/BaikeLemmaCardApi"
        params = {
            'scope': '103',
            'format': 'json',
            'appid': '379020',
            'bk_key': keyword,
            'bk_length': '600'
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and 'abstract' in data:
                    print(f"  ✅ 获取成功！")
                    return {
                        'title': data.get('title', keyword),
                        'summary': data.get('abstract', ''),
                        'url': data.get('url', '')
                    }
        except:
            pass
        
        print(f"  ⚠️  API获取失败")
        return None
    
    def interactive_enhance(self, base_data):
        """交互式增强内容"""
        print("\n" + "="*60)
        print("📝 内容增强")
        print("="*60)
        
        if base_data:
            print(f"\n已获取基础信息:")
            print(f"标题: {base_data['title']}")
            print(f"摘要: {base_data['summary'][:100]}...")
        else:
            print("\n未获取到API数据，需要手动输入")
        
        print("\n" + "="*60)
        print("现在需要你帮忙补充详细信息")
        print("="*60)
        
        # 引导用户
        sections = {
            '学院设置': '请列出主要学院（如：建筑学院、文学院...）',
            '世界排名': '请输入最新的QS/THE排名',
            '招生信息': '请输入学费、招生要求等',
            '校园生活': '请输入住宿、社团等信息',
        }
        
        content = {}
        
        print("\n💡 提示：可以")
        print("  1. 从浏览器复制相关段落")
        print("  2. 直接输入简要信息")
        print("  3. 输入'skip'跳过某部分")
        print()
        
        for section, hint in sections.items():
            print(f"\n{'='*40}")
            print(f"📌 {section}")
            print(f"   {hint}")
            print(f"{'='*40}")
            print("请输入内容（完成后单独一行输入 END）:")
            
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip().upper() == 'END':
                        break
                    if line.strip().lower() == 'skip':
                        print("  ⏭️  已跳过")
                        break
                    lines.append(line)
                except EOFError:
                    break
            
            text = '\n'.join(lines)
            if text.strip() and text.strip().lower() != 'skip':
                content[section] = text
        
        return content
    
    def build_document(self, base_data, sections):
        """构建完整文档"""
        doc = []
        
        # 标题
        if base_data:
            doc.append(f"# {base_data['title']}\n")
            doc.append(f"来源: API + 人工补充")
            doc.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            doc.append("---\n")
            
            # 摘要
            if base_data.get('summary'):
                doc.append("## 概述\n")
                doc.append(base_data['summary'] + "\n")
        else:
            doc.append(f"# 香港大学\n")
            doc.append(f"来源: 人工整理")
            doc.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            doc.append("---\n")
        
        # 各个章节
        for section, content in sections.items():
            doc.append(f"\n## {section}\n")
            doc.append(content + "\n")
        
        return '\n'.join(doc)
    
    def save(self, content, filename="enhanced_hku.txt"):
        """保存文件"""
        filepath = os.path.join(self.kb_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n{'='*60}")
        print(f"✅ 已保存到: {filepath}")
        print(f"📊 总长度: {len(content)} 字符")
        print(f"{'='*60}")
        
        return filepath
    
    def quick_mode(self):
        """快速模式 - 一次性输入"""
        print("\n📝 快速模式：直接粘贴所有内容")
        print("="*60)
        print("从浏览器复制整个百度百科页面，粘贴到这里")
        print("输入完成后，单独一行输入 END")
        print("="*60)
        
        lines = []
        print()
        while True:
            try:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            except EOFError:
                break
        
        content = '\n'.join(lines)
        
        if len(content) < 100:
            print("⚠️  内容太少")
            return None
        
        # 简单清洗
        # 移除常见导航文字
        noise = ['百度首页', '登录', '网页', '新闻', '贴吧', '知道', '视频', 
                '音乐', '图片', '地图', '文库', '更多', '搜索', '编辑']
        
        lines = content.split('\n')
        cleaned = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:
                continue
            if any(n in line for n in noise):
                continue
            cleaned.append(line)
        
        content = '\n\n'.join(cleaned)
        
        # 构建文档
        doc = f"""# 香港大学

来源: 百度百科（人工收集）
整理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

---

{content}
"""
        
        return doc


def main():
    print("\n" + "="*60)
    print("🎯 智能知识收集器")
    print("="*60)
    
    print("\n选择模式:")
    print("  1. 智能模式（API + 分段输入）- 推荐")
    print("  2. 快速模式（一次性粘贴）")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    collector = SmartCollector()
    
    if choice == '2':
        # 快速模式
        content = collector.quick_mode()
        if content:
            collector.save(content, "quick_hku.txt")
            print("\n✅ 完成！记得重启Agent")
        else:
            print("\n❌ 失败")
    
    else:
        # 智能模式
        print("\n🤖 智能模式")
        
        # 1. 获取API数据
        base_data = collector.fetch_api_summary('香港大学')
        
        # 2. 交互式增强
        sections = collector.interactive_enhance(base_data)
        
        # 3. 构建文档
        doc = collector.build_document(base_data, sections)
        
        # 4. 保存
        collector.save(doc, "smart_hku.txt")
        
        print("\n✅ 完成！记得重启Agent")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")