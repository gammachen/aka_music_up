# test_echarts_data.py
import json
import sys
import os

# 添加父目录到系统路径，以便导入read_a_book_agent模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.read_a_book_agent import generate_relations_json_for_echarts

def main():
    # 调用函数生成ECharts数据
    knowledge_path = "/Users/shhaofu/Code/cursor-projects/aka_music/knowledge.json"
    echarts_data = generate_relations_json_for_echarts(knowledge_path)
    
    # 打印节点和连接数量
    print(f"生成了 {len(echarts_data['nodes'])} 个节点和 {len(echarts_data['links'])} 个连接")
    
    # 保存到文件
    output_path = "/Users/shhaofu/Code/cursor-projects/aka_music/echarts_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(echarts_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到 {output_path}")

if __name__ == "__main__":
    main()