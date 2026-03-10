import json

# 交换对
swap_pairs = [(10, 218), (77, 231), (110, 232), (149, 250)]

# 读取原json
with open('../data/svamp.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 建立 index 到对象的映射（便于查找和修改）
index2obj = {item['index']: item for item in data}

# 执行交换
for idx1, idx2 in swap_pairs:
    # 交换index
    if idx1 in index2obj and idx2 in index2obj:
        index2obj[idx1]['index'], index2obj[idx2]['index'] = idx2, idx1

# 重新生成list并按index排序
data_sorted = sorted(data, key=lambda x: x['index'])

# 保存到新的json文件
with open('svamp.json', 'w', encoding='utf-8') as f:
    json.dump(data_sorted, f, ensure_ascii=False, indent=2)

print("Done! Saved to output.json.")
