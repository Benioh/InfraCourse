# 系统地图：L08 Megatron `.bin/.idx` 数据预处理

L08 进入训练数据主线。前面已经讲过训练 step、显存账本、分布式同步、kernel 和 activation checkpoint；这些都假设训练 loop 能稳定拿到 token 序列。本讲补上这条输入链：原始文本怎样离线变成 Megatron 可以 mmap 读取的 IndexedDataset。

## 1. Megatron bin/idx 数据预处理系统图

![系统地图：L08 Megatron `.bin/.idx` 数据预处理：Megatron bin/idx 数据预处理系统图](outputs/system-map-01.png)

系统图把原始文本从训练 loop 前移到离线预处理：tokenizer 先把文本变成 token，`.bin` 连续保存 token，`.idx` 保存样本偏移和长度，训练时按索引随机访问。

## 2. bin、idx、dtype、随机访问概念图

![系统地图：L08 Megatron `.bin/.idx` 数据预处理：bin、idx、dtype、随机访问概念图](outputs/concept-map-01.png)

概念依赖是 `.idx` 先描述边界，`.bin` 再承载连续 token；dtype 决定存储空间和可表示范围；读取器必须按 offset/length 切片，不能在训练 step 里重新做原始文本处理。

## 3. 本课边界

- patch 使用简化 idx 格式，不复刻完整 Megatron indexed dataset。
- 预处理结论要记录 tokenizer、dtype、样本数和文件大小。
- 训练吞吐问题要先区分离线数据格式和在线 dataloader。
