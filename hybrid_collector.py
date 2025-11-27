#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合知识收集工具 - 简单快速
"""

import os
import time
import webbrowser

def collect():
    kb_dir = "knowledge_base"
    os.makedirs(kb_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("📚 知识收集工具")
    print("="*60)
    
    sources = [
        ("百度百科-香港大学", "https://baike.baidu.com/item/香港大学/216819"),
        ("知乎-香港大学话题", "https://www.zhihu.com/topic/19558464/hot"),
        ("自定义", "")
    ]
    
    print("\n选择来源：")
    for i, (name, url) in enumerate(sources, 1):
        print(f"  {i}. {name}")
    
    choice = input("\n输入编号: ").strip()
    
    if choice == '3':
        url = input("输入URL: ").strip()
        name = input("页面主题: ").strip()
    else:
        try:
            idx = int(choice) - 1
            name, url = sources[idx]
        except:
            print("无效选择")
            return
    
    # 打开浏览器
    if url:
        print(f"\n🌐 正在打开: {url}")
        webbrowser.open(url)
    
    print("\n" + "="*60)
    print("📋 操作步骤：")
    print("1. 在浏览器中复制内容")
    print("2. 回到这里粘贴")
    print("3. 单独一行输入 END 结束")
    print("="*60)
    
    input("\n按回车继续...")
    
    print("\n粘贴内容（完成后输入END）：\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
    
    content = '\n'.join(lines)
    
    if len(content) < 100:
        print("\n⚠️  太短了")
        return
    
    # 保存
    topic = name.replace(' ', '_').replace('-', '_')
    filename = f"manual_{topic}.txt"
    filepath = os.path.join(kb_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {name}\n\n")
        f.write(f"来源: {url}\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("---\n\n")
        f.write(content)
    
    print(f"\n✅ 已保存: {filepath}")
    print(f"📊 {len(content)} 字符")
    
    if input("\n继续添加? (y/n): ").lower() == 'y':
        collect()

if __name__ == "__main__":
    try:
        collect()
    except KeyboardInterrupt:
        print("\n\n退出")