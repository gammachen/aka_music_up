import dedupe

    
def main():
    print("知识图谱构建工具")

    # 示例数据，补充到10条以上
    # 假设字段为 Email, Phone, Company

    data = {
        1: {"Email": "alice@example.com", "Phone": "1234567890", "Company": "Acme"},
        2: {"Email": "bob@example.com", "Phone": "1234567891", "Company": "Acme"},
        3: {"Email": "alice@example.com", "Phone": "1234567890", "Company": "Acme Corp"},
        4: {"Email": "carol@example.com", "Phone": "1234567892", "Company": "Beta"},
        5: {"Email": "dave@example.com", "Phone": "1234567893", "Company": "Beta"},
        6: {"Email": "eve@example.com", "Phone": "1234567894", "Company": "Gamma"},
        7: {"Email": "frank@example.com", "Phone": "1234567895", "Company": "Gamma"},
        8: {"Email": "grace@example.com", "Phone": "1234567896", "Company": "Delta"},
        9: {"Email": "heidi@example.com", "Phone": "1234567897", "Company": "Delta"},
        10: {"Email": "ivan@example.com", "Phone": "1234567898", "Company": "Epsilon"},
        11: {"Email": "judy@example.com", "Phone": "1234567899", "Company": "Epsilon"},
    }

    # 字段定义
    fields = [
        dedupe.variables.Exact("Email"),
        dedupe.variables.String("Phone"),
        dedupe.variables.String("Company", has_missing=True),
    ]

    # 初始化
    deduper = dedupe.Dedupe(fields)
    # deduper.sample(data, 20)
    deduper.prepare_training(data)


    # 这里用人工标注模拟，实际可用 deduper.consoleLabel() 进行交互式标注
    # 假设我们已经有训练数据
    # labeled_examples 示例，需与data中的ID对应
    labeled_examples = {
        'match': [
            (data[1], data[3]), # alice, Acme vs alice, Acme Corp
            (data[4], data[5]),  # carol, Beta vs dave, Beta
            (data[6], data[7]), # eve, Gamma vs frank, Gamma
            (data[8], data[9]),  # grace, Delta vs heidi, Delta
            (data[10], data[11]),  # ivan, Epsilon vs judy, Epsilon
        ],
        'distinct': [
            (data[1], data[2]),  # alice vs bob
            (data[1], data[4]),  # alice vs carol
            (data[2], data[6]),  # bob vs eve
            (data[3], data[8]),  # alice vs grace
            (data[5], data[10]), # dave vs ivan
            (data[7], data[11]), # frank vs judy
        ]
    }
    # 后续可用 deduper.mark_pairs(labeled_examples) 进行自动标注
    deduper.mark_pairs(labeled_examples)
    deduper.train()

    # 阈值可调，0.5~0.7
    threshold = 0.6
    # clustered_dupes = deduper.match(data, threshold)
   
    print("去重结果：")
    clusters = deduper.partition(data, threshold)
    for cluster in clusters:
        record_ids, scores = cluster  # cluster 是 (tuple_of_ids, array_of_scores)
        print("Cluster:", record_ids, "Scores:", scores)
        for record_id in record_ids:
            print(data[record_id])

if __name__ == '__main__':
    main()    
    
'''
(translate-env) shhaofu@shhaofudeMacBook-Pro p-llm-knowledge % python 13_dedupe_etl_crm.py
知识图谱构建工具
去重结果：
Cluster: (1, 2) Scores: [0.6455611 0.6455611]
{'Email': 'alice@example.com', 'Phone': '1234567890', 'Company': 'Acme'}
{'Email': 'bob@example.com', 'Phone': '1234567891', 'Company': 'Acme'}
Cluster: (4, 5) Scores: (0.6455611, 0.6455611)
{'Email': 'carol@example.com', 'Phone': '1234567892', 'Company': 'Beta'}
{'Email': 'dave@example.com', 'Phone': '1234567893', 'Company': 'Beta'}
Cluster: (6, 7) Scores: (0.6455611, 0.6455611)
{'Email': 'eve@example.com', 'Phone': '1234567894', 'Company': 'Gamma'}
{'Email': 'frank@example.com', 'Phone': '1234567895', 'Company': 'Gamma'}
Cluster: (8, 9) Scores: (0.6455611, 0.6455611)
{'Email': 'grace@example.com', 'Phone': '1234567896', 'Company': 'Delta'}
{'Email': 'heidi@example.com', 'Phone': '1234567897', 'Company': 'Delta'}
Cluster: (10, 11) Scores: (0.6455611, 0.6455611)
{'Email': 'ivan@example.com', 'Phone': '1234567898', 'Company': 'Epsilon'}
{'Email': 'judy@example.com', 'Phone': '1234567899', 'Company': 'Epsilon'}
Cluster: (3,) Scores: (1.0,)
{'Email': 'alice@example.com', 'Phone': '1234567890', 'Company': 'Acme Corp'}

你的去重结果如下（每个Cluster后面是成员ID和分数）：

```
Cluster: (1, 2) Scores: [0.6455611 0.6455611]
{'Email': 'alice@example.com', 'Phone': '1234567890', 'Company': 'Acme'}
{'Email': 'bob@example.com', 'Phone': '1234567891', 'Company': 'Acme'}
Cluster: (4, 5) Scores: (0.6455611, 0.6455611)
{'Email': 'carol@example.com', 'Phone': '1234567892', 'Company': 'Beta'}
{'Email': 'dave@example.com', 'Phone': '1234567893', 'Company': 'Beta'}
Cluster: (6, 7) Scores: (0.6455611, 0.6455611)
{'Email': 'eve@example.com', 'Phone': '1234567894', 'Company': 'Gamma'}
{'Email': 'frank@example.com', 'Phone': '1234567895', 'Company': 'Gamma'}
Cluster: (8, 9) Scores: (0.6455611, 0.6455611)
{'Email': 'grace@example.com', 'Phone': '1234567896', 'Company': 'Delta'}
{'Email': 'heidi@example.com', 'Phone': '1234567897', 'Company': 'Delta'}
Cluster: (10, 11) Scores: (0.6455611, 0.6455611)
{'Email': 'ivan@example.com', 'Phone': '1234567898', 'Company': 'Epsilon'}
{'Email': 'judy@example.com', 'Phone': '1234567899', 'Company': 'Epsilon'}
Cluster: (3,) Scores: (1.0,)
{'Email': 'alice@example.com', 'Phone': '1234567890', 'Company': 'Acme Corp'}
```

---

## 你的 labeled_examples 内容（正负样本）

**match（认为是同一实体）**
- (1, 3)  # alice, Acme vs alice, Acme Corp
- (4, 5)  # carol, Beta vs dave, Beta
- (6, 7)  # eve, Gamma vs frank, Gamma
- (8, 9)  # grace, Delta vs heidi, Delta
- (10, 11)  # ivan, Epsilon vs judy, Epsilon

**distinct（认为不是同一实体）**
- (1, 2)  # alice vs bob
- (1, 4)  # alice vs carol
- (2, 6)  # bob vs eve
- (3, 8)  # alice vs grace
- (5, 10) # dave vs ivan
- (7, 11) # frank vs judy

---

## 分析

### 1. **聚类结果与labeled_examples的match部分对比**

- (4, 5), (6, 7), (8, 9), (10, 11) 这四组完全聚到了一起，和你的match标注一致。
- (1, 2) 被聚到一起，但你在labeled_examples中标注为distinct（即你认为1和2不是同一实体），但模型聚到了一起。
- (1, 3) 没有聚到一起，3单独成一组，但你标注为match（即你认为1和3是同一实体），但模型没有聚到一起。

### 2. **distinct部分对比**

- (1, 2) 被聚到一起，和你的distinct标注**不一致**（模型认为是同一实体，你认为不是）。
- 其他distinct对没有被聚到一起，和你的标注一致。

### 3. **单独成组的情况**

- (3,) 单独成组，说明模型认为它和其他任何一条都不像（但你标注它和1是match）。

---

## 结论

- **大部分聚类结果和你的labeled_examples一致**，如(4,5)、(6,7)、(8,9)、(10,11)。
- **有两处不一致**：
    - (1, 2) 被聚到一起，但你标注为distinct。
    - (1, 3) 没有聚到一起，但你标注为match。
- 这说明模型在区分(1,2,3)这三条时，特征区分度不够，或者训练样本还不够多/不够典型。

---

## 建议

1. **增加更多正负样本，尤其是1、2、3相关的样本**，让模型更好地学习区分。
2. **检查字段特征**，如Email、Phone、Company是否足够区分。
3. **调低阈值**（threshold），有时可以让更多“模糊匹配”分开。

---
'''