# 改进方案技术报告：从原始版本到 cem-fixed 的架构演进

## 一、背景与出发点

本工作的目标是在分割学习（Split Learning）场景下，通过对编码器施加正则化约束来防御模型逆向攻击（Model Inversion Attack，MIA）。防御的核心直觉来自 CEM（Cluster Entropy Maximization）思想：若同一类别的样本在特征空间中形成多个紧密的子簇，而非均匀弥散，则攻击者从中间层特征重建原始图像的难度将大幅提升。为了将这一思想与可微分的注意力机制相结合，我们在根目录中设计了第一版实现，命名为 SlotCrossAttentionCEM。然而经过仔细分析，该版本存在三个根本性的逻辑错误，导致其不仅无法实现预期的防御效果，甚至可能在训练过程中产生反向效果。cem-fixed 版本针对这三个错误逐一进行了重构，同时对模块架构和工程实现做了若干关键改进。

---

## 二、第一个核心错误：损失函数的优化方向相反

原始版本在计算正则化损失时，使用了如下公式：

```python
per_sample = F.relu(torch.log(threshold_t) - torch.log(var_per_slot + eps))
```

这个公式的含义是：当 `var_per_slot` 低于阈值 `threshold` 时，损失为正，惩罚"方差太小"的情况；当方差高于阈值时，损失为零，不施加任何约束。换言之，这条损失在驱使编码器**增大**类内方差，让同类特征尽量分散。这与 CEM 的防御目标完全相反。CEM 的正确目标是迫使编码器将同类特征**压缩**进若干个紧密子簇，使每个子簇内部的方差保持在阈值以下；一旦某个子簇的方差超出阈值，才施加惩罚促使收敛。

cem-fixed 将损失方向修正为：

```python
per_slot_loss = F.relu(torch.log(var_per_slot + γ) - torch.log(threshold))
```

该公式在 `var_per_slot` 超出阈值时损失为正，低于阈值时为零，梯度信号的方向指向"减小方差"，才真正符合 CEM 的防御意图。值得注意的是，阈值计算公式也与 CEM-main 保持一致，即 `threshold = var_threshold × reg_strength² + γ`，其中 `γ=0.01` 是数值稳定项，从而在实验参数可调的同时保证了方法的一致性。

---

## 三、第二个核心错误：方差的来源是神经网络输出而非几何计算

原始版本设计了一个 `head_logvar` 预测头，结构为一层线性层加 Softplus 激活，对每个 slot 的表示直接预测一个"方差向量"：

```python
self.head_logvar = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Softplus())
var_s = self.head_logvar(slots) + eps_var
```

这种设计将"方差"视为一个可以被神经网络自由预测的数值，而非特征空间中样本到质心的真实几何距离。其后果是：在反向传播时，网络完全可以通过调整 `head_logvar` 的参数来输出满足损失要求的数值，而对编码器本身的特征分布不产生任何约束。也就是说，正则化的压力被 `head_logvar` 这一"缓冲层"完全吸收，编码器的梯度几乎不受 CEM loss 的影响，防御效果等同于没有施加正则化。

cem-fixed 彻底删除了 `head_logvar` 和 `head_mu`，改为用几何方式直接计算每个 slot 对应的加权方差：

```python
centroids = einsum(attn_norm, feats_input).detach()        # 加权质心，阻断梯度
diff = feats_input.unsqueeze(2) - centroids.unsqueeze(1)   # 样本到质心的差向量
sq_diff = diff ** 2
var_per_slot = einsum(attn_norm, sq_diff) / slot_dim       # 注意力加权方差
```

这里的方差完全由当前 batch 中样本特征的空间位置决定，与神经网络参数无关，因此损失的梯度会真实地反映编码器特征的几何分布状况，并通过反向传播直接作用于编码器。

---

## 四、第三个核心错误：Slot Attention 的聚类对象完全错误

这是三个错误中语义偏差最大的一个。原始版本在处理输入特征时，先将形状为 `[B, feature_dim]` 的 batch 特征扩展为 `[B, 1, feature_dim]`，即每张图像作为独立的输入送入 Slot Attention：

```python
if features.dim() == 2:
    features = features.unsqueeze(1)   # [B, 1, D]
slots = self.slot_attention(features)  # 每张图独立处理
```

在这种输入形状下，Slot Attention 每次只看到单张图像的一个 token，用 8 个 slot 对 1 个 token 做竞争性聚类，在数学上没有任何意义。Slot Attention 的设计初衷是让多个 slot 竞争地"捕获"输入序列中的不同子集，当输入只有 1 个 token 时，8 个 slot 会全部收敛到同一个位置，聚类退化为恒等变换。更根本的问题在于，CEM 所需的"类内聚类"本质上是一个**跨样本**的操作——需要把同一类别的多个样本作为一组，在它们构成的特征集合上发现若干紧致子群。原始版本的实现让每张图像在自身内部独立进行"聚类"，与 CEM 的语义完全不符。

cem-fixed 将聚类操作的粒度从"单张图"更正为"同类多个样本"。对于每个类别 `c`，提取当前 batch 中所有属于该类的样本特征 `batch_feats = z_proj[labels == c]`，形成一个 `[N_c, slot_dim]` 的序列，将其作为 Slot Attention 的输入 `feats_input = all_feats.unsqueeze(0)` 即 `[1, N_c, slot_dim]`。这样，8 个 slot 竞争地对 `N_c` 个样本进行软聚类，每个 slot 代表一个发现的子群，输出的注意力权重 `attn_weights: [1, N_c, S]` 表示每个样本属于每个子群的概率，这才是 CEM 在特征空间中所需要的语义。

---

## 五、辅助改进一：引入 FeatureMemoryBank 解决跨样本信息不足的问题

将聚类对象更正为"同类多个样本"后，随之而来的工程问题是：在标准训练设置下（batch size=64，共530个类），每个 batch 中平均每类只有约1到2个样本，数量远不足以支撑稳定的聚类。当 `all_feats.size(0) < 2` 时，该类会被直接跳过，大量类别无法参与 CEM 优化。

为解决这一问题，cem-fixed 引入了 `FeatureMemoryBank`，这是一个按类别索引的环形缓冲区，每类存储最近64个经过投影后的特征向量。每个 training step 中，当前 batch 的新特征会以 `no_grad` 方式写入对应类别的槽位；在计算 CEM loss 时，从 bank 中读取该类的历史特征（无梯度），与当前 batch 的特征（有梯度）拼接后再进行聚类。这样即便某类在当前 batch 中只出现1次，也能借助 bank 中积累的历史样本形成一个规模足够的特征集合，保证每类都有有效的 CEM 梯度信号。需要特别说明的是，FeatureMemoryBank 是一个普通 Python 类而非 `nn.Module`，因此需要在 `SlotCrossAttentionCEM.to()` 方法中显式覆盖，将 bank 内的所有 tensor 随模型一起迁移至 GPU，避免在 forward 计算中出现 device 不一致的错误。

---

## 六、辅助改进二：引入 proj_down 解决高维特征与参数量的矛盾

在 FaceScrub 数据集上，VGG11-BN-SGM 编码器在 cutlayer=4 处输出形状为 `[B, 8, 16, 16]`，展平后特征维度为2048。原始版本的 Slot Attention 直接在特征的原始维度上运行（因为 slot_dim 默认等于 feature_dim），导致模型中 `to_q`、`to_k`、`to_v` 等线性层的参数量约为 `3 × 2048 × 2048 ≈ 1260万`，GRU 参数量更高，整个 SlotCrossAttentionCEM 模块参数量超过1亿，占用大量显存并严重拖慢训练速度。

cem-fixed 在输入 Slot Attention 之前增加了一个线性投影层 `proj_down = nn.Linear(2048, slot_dim)`，将特征维度从2048降至128（或其他可配置的值），Slot Attention 全程在128维的投影空间中操作，参数量降至约100万。梯度仍然可以通过 `proj_down` 流回编码器的全部2048个维度，防御效果不受影响，同时使实验在合理的时间内完成成为可能。此外，`proj_down` 的存在也使聚类空间与特征空间解耦，Slot Attention 可以专注于学习适合聚类的低维表示，而不受原始高维特征中噪声维度的干扰。

---

## 七、辅助改进三：detach 质心以避免梯度"作弊"

在 cem-fixed 的几何方差计算中，质心通过注意力权重对特征的加权平均得到：

```python
centroids = einsum(attn_norm, feats_input).detach()
```

此处对质心显式施加了 `.detach()`。若不 detach，梯度可以同时流向"features 向质心移动"和"质心向 features 移动"两条路径。在后者存在的情况下，网络存在一条捷径：通过调整注意力权重让质心直接追随特征，而无需让特征真正收敛，从而以极小的实际约束满足损失要求。detach 之后，质心在反向传播中是固定的参考点，梯度唯一的来源是 `(features - detached_centroid)²` 对 features 的导数，这才保证了编码器的特征必须真实地向质心靠拢，而非通过质心漂移来规避惩罚。

---

## 八、辅助改进四：推理阶段的确定性

原始版本的 Slot Attention 在所有情况下都使用带噪声的随机初始化：

```python
slots = mu + sigma * torch.randn_like(mu)  # 训练和推理都随机
```

cem-fixed 区分了训练和推理两种模式：

```python
if self.training:
    slots = mu + sigma * torch.randn_like(mu)   # 训练：随机初始化，增加探索性
else:
    slots = mu.clone()                           # 推理/测试：确定性，结果可复现
```

在测试阶段运行 MIA 评估时，随机的 slot 初始化会导致每次 forward 产生不同的注意力分布，进而导致 CEM loss 的统计量出现波动，不利于公平比较和结果复现。推理时使用确定性初始化消除了这一随机性来源。

---

## 九、总结

综上所述，原始版本与 cem-fixed 之间的差异并非参数调优层面的改进，而是针对三个导致方法完全失效的根本性错误的全面重构。损失方向的修正确保了梯度信号的语义正确；将方差计算从神经网络预测改为几何计算确保了梯度能够真实作用于编码器；将聚类粒度从单样本内部更正为跨样本操作确保了 CEM 的防御语义得以实现。在此基础上，FeatureMemoryBank 解决了跨样本聚类所必需的样本数量问题，proj_down 解决了高维特征的参数效率问题，centroid detach 堵住了梯度优化中的捷径，推理确定性保证了评估的可靠性。这些改动共同构成了一个在理论层面自洽、在工程层面可靠运行的完整防御方法。
