import os
from poem_analyzer import process_poems_by_author

# 处理李白的诗歌
author_name = "李白"
poems_file_path = process_poems_by_author(author_name)

print(f"已将李白的诗歌保存到 {poems_file_path}")

# 如果需要使用智谱AI丰富诗歌信息，可以取消下面的注释并提供API密钥
# from poem_analyzer import enrich_poems_with_ai
# api_key = "your_zhipu_api_key_here"  # 替换为实际的智谱AI API密钥
# enriched_file_path = enrich_poems_with_ai(poems_file_path, api_key)
# print(f"已将丰富后的诗歌保存到 {enriched_file_path}")