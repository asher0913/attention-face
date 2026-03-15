# 组会讲稿：SCA-CEM 超参数扫描实验

> 对应 `run3slides/main.pdf`，共 19 页。每一节对应一页 slide。

---

## 第 1 页 — 封面

大家好，今天给大家汇报一下我在 Split Learning 隐私防御方向上的最新实验进展。题目是 SlotCrossAttention-CEM 的超参数扫描实验，目标数据集是 FaceScrub，530 个人脸身份。

---

## 第 2 页 — Overview: What Are We Doing?

首先说一下我们在做什么。

我们的任务是防御 Split Learning 中的模型逆向攻击（Model Inversion Attack, MIA）。攻击者能拿到客户端发给服务器的中间特征（smashed data），然后训练一个解码器把原始人脸图片还原出来。我们要做的就是让这个还原尽可能失败。

衡量标准是推理阶段的 SSIM 值——越低说明攻击者重建的图像越差，也就是我们的防御越好。原始论文（CEM-main 方法）在 FaceScrub 上的 baseline inference SSIM 大约是 0.532。

我们的创新是把原始论文里的硬聚类（KMeans）替换成了一个**可微分的 Slot Attention 机制**，叫做 SlotCrossAttentionCEM。好处是端到端可训练，不需要每个 epoch 对整个数据集做一次 KMeans 重新聚类。

这次实验一共设计了 **20 组实验**，分成 A 到 F 六个组，系统地扫描关键超参数，争取找到一组配置能打败原始论文的 SOTA 结果。

---

## 第 3 页 — Architecture Pipeline: 8 Stages

这张图是我们整个算法的流水线，一共 8 个阶段。

上面一行是前向传播主路径：
- **Stage 1**：客户端编码器，把输入图片编码成 smashed data z
- **Stage 2**：给 z 加高斯噪声，得到 z_n
- **Stage 3**：Slot Attention——对 z_n 做可微分的软聚类
- **Stage 4**：Cross Attention——用原始特征精细化 slot 表示

下面一行是损失计算和优化：
- **Stage 5**：从 slot 向量预测每个聚类的方差
- **Stage 6**：计算 CEM 损失——要求每个身份类内的方差足够大
- **Stage 7**：标准交叉熵分类损失
- **Stage 8**：双反向传播——先算 CEM 梯度保存下来，再算 CE 梯度，最后手动把 CEM 梯度加回编码器

图下面标注了每个阶段受哪些超参数控制。另外还有两个跨阶段的控制参数：warmup 控制 Stage 3-6 什么时候激活，epochs 控制总训练时长。

---

## 第 4 页 — Stage 1: Encoder

Stage 1 是编码器，在所有 20 组实验中**完全不变**。

架构是 VGG11 带 BatchNorm，在第 4 层做模型切分。切分点之后有一个瓶颈层，把通道数压缩到 8（noRELU_C8S1 配置），然后经过一个 LCA 稀疏编码层（Locally Competitive Algorithm），最后用 Sigmoid 激活函数。

最终输出的 smashed data z 的维度是 8×8×8（8 个通道、8×8 的空间尺寸）。

为什么用 Sigmoid 而不是 ReLU？因为 Sigmoid 的输出范围是 [0,1]，可以限制传输给服务器的信息量，而且和后面的加噪声操作更兼容。

这个阶段没有可调参数，20 组实验完全一致。

---

## 第 5 页 — Stage 2: Gaussian Noise Injection

Stage 2 是高斯噪声注入，受 **noise**（σ）参数控制。

公式很简单：z_n = z + N(0, σ²)。注意，这个噪声在**训练和推理阶段都会加**，是对传输通道的保护。

增大 noise 有三个效果：
1. **正面**：攻击者的解码器要先去噪才能重建，难度增大
2. **正面**：CEM 的阈值会自动提高——因为阈值的计算公式是 threshold = var_threshold × σ²，所以 σ 增大阈值也跟着增大
3. **负面**：分类器收到的信号更noisy，可能会降低分类准确率

所以 noise 有一个**双重作用**：它既在 Stage 2 保护传输通道，又在 Stage 6 抬高 CEM 的方差门槛。

各实验的设置可以看右边的表格。baseline 是 0.05，我们的实验从 0.05 到 0.10 不等。比如 exp07 用了 0.10，是 baseline 的 2 倍，属于最激进的设置。

---

## 第 6 页 — Stage 3: Slot Attention

Stage 3 是我们的核心创新——Slot Attention 软聚类，取代了原始论文中的 KMeans 硬聚类。

算法流程：
1. 初始化 S 个 slot 原型向量，从可学习的均值和方差中采样
2. 迭代 K 次（K 就是 iters 参数）：
   - 用 slot 做 query，用特征 z 做 key 和 value，算注意力权重
   - 对注意力权重做归一化——这一步类似竞争性分配，让每个特征点只被少数 slot 关注
   - 用注意力加权的 value 更新 slot
   - 经过一个 GRU 门控循环单元，融合旧的 slot 和新的更新
   - 再过一层 MLP 做残差连接

受两个参数控制：
- **slots**：slot 原型的数量。baseline 是 8，我们测了 4 到 16。太少（4 个）就是太粗的聚类，530 个身份只用 4 个原型来代表；太多（16 个）则可能出现 slot collapse——有些 slot 永远不会被分配到任何特征，变成"死"slot。
- **iters**：GRU 迭代次数。baseline 是 3 次，我们测了 3 到 5。更多迭代让 slot 竞争更充分，但增加计算量，而且可能出现梯度消失。

---

## 第 7 页 — Stage 4: Cross Attention

Stage 4 用 Cross Attention 对 slot 做进一步精细化，受 **heads** 参数控制。

机制是多头注意力：slot 向量做 query，原始特征 z 做 key 和 value。这样每个 slot 可以重新参考原始特征来修正自己。

一个重要的设计是 **Tanh 门控**：
- 输出 = slots + tanh(α) × CrossAttn(slots, z)
- 再经过 FFN：输出 += tanh(β) × FFN(输出)

α 和 β 是可学习的标量，初始值 0.1。tanh(0.1) ≈ 0.1，所以训练初期 cross attention 的贡献被大幅抑制。随着训练推进，模型会逐渐"开门"让 cross attention 发挥作用。这样避免了训练初期 cross attention 的噪声信号干扰 slot 的收敛。

heads 必须能整除特征维度 8，所以只能取 1、2、4、8。baseline 是 4，只有 exp18 改成了 2。heads=2 意味着每个 head 看 4 维的子空间（更宽的视角），而 heads=4 每个 head 只看 2 维。

---

## 第 8 页 — Stage 5: Variance Prediction

Stage 5 从 slot 向量预测方差，**没有可调超参数**。

公式是：
- μ = head_mu(slots) = 线性层
- var = Softplus(head_logvar(slots)) + ε

关键设计：方差是**神经网络预测**的，不是像 KMeans 那样通过几何距离计算的。Softplus 保证方差始终为正。还有 EMA 运行统计量来稳定训练。

跟原始 CEM 对比——右边这个表很重要：
- 原始方法：方差 = 样本到聚类中心的平方距离的均值，是几何计算，**不可微分**，而且是上一个 epoch 算的（滞后一个 epoch）
- 我们的方法：方差 = Softplus(W·slots)，是学习出来的表示，**完全可微分**，**实时**计算

这个阶段虽然没有可调参数，但它的质量完全依赖于上游 Stage 3 和 4 的 slots、iters、heads 设置。

---

## 第 9 页 — Stage 6: CEM Loss Computation

Stage 6 是 CEM 损失的计算，受 **var_threshold** 和 **loss_scale** 两个参数控制。

损失公式：对每个身份类、每个样本，计算 ReLU(log(threshold) - log(predicted_var))。意思是：如果预测的方差已经超过了阈值，这项就是 0（loss 为 0，满足要求了）；如果方差低于阈值，就产生正的损失，推动编码器增大特征的分散程度。

阈值的计算：threshold = var_threshold × noise² + ε。所以阈值由两个参数共同决定。

var_threshold 从 baseline 的 0.15 提高到最大 0.25。noise 从 0.05 提高到最大 0.10。两者相乘后，实际阈值从 baseline 的 3.75e-4 最高到了 2.00e-3，提高了 5 倍多。

loss_scale 是在反向传播之前对 CEM loss 的缩放系数。**baseline 的一个关键问题**就是 loss_scale=0.5，等于把 CEM loss 直接减半了，这导致防御力度不足。我们所有 20 组实验都把它修正到了 0.8 以上。

---

## 第 10 页 — Stages 7 & 8: Classification + Double Backward

Stage 7 是标准的分类损失，没有改动。

Stage 8 是训练的核心——双反向传播。流程是：
1. 先对 CEM loss 做 backward（保留计算图）
2. 保存编码器参数的 CEM 梯度
3. 清零优化器梯度
4. 对分类 CE loss 做 backward
5. 手动把保存的 CEM 梯度乘以 lambd 加回编码器参数

这里还有一个自适应缩放机制：当学习率大于 0.00041 时（也就是训练早期），CEM 梯度会被缩放为 0.001/lr，大幅压制；只有当学习率降到 0.00041 以下时（训练后期），CEM 才达到全功率。这个设计让模型先学好分类，后期再全力做防御。

**有效 CEM 强度 = lambd × loss_scale**。baseline 只有 16 × 0.5 = 8.0，我们的实验从 24.0 推到了最高 43.2（exp20），是 baseline 的 5.4 倍。

---

## 第 11 页 — Cross-Stage Control: warmup and epochs

两个跨阶段控制参数：

**warmup**：CEM 激活前的预热期。在 warmup 期间，Stage 3 到 6 完全跳过，模型只做纯分类训练。
- baseline warmup=3，也就是前 3 个 epoch 只训分类
- exp12 设成 1，几乎立刻开始 CEM
- exp13、14 设成 5，让分类器先充分收敛

**epochs**：总训练轮数。由于 CosineAnnealing 学习率调度的存在，CEM 只在训练最后约 5% 的 epoch 才达到全功率。所以增加总 epoch 数就能延长 CEM 全功率的时间窗口。
- baseline 240 epoch：约 12 个 epoch 的全功率 CEM
- 大部分实验 300 epoch：约 15 个 epoch
- exp19 用 360 epoch：约 18 个 epoch，比 baseline 多了 50%

---

## 第 12 页 — Experiment Groups: Design Rationale

我们的实验分成 6 组，每组有明确的研究问题：

- **A 组（exp01-03）**：只改 CEM 强度（lambd 和 loss_scale），其他全不变。回答的问题是：单纯增大 CEM 梯度注入量能不能改善防御？
- **B 组（exp04-07）**：推高噪声 noise 和方差阈值 var_threshold。回答：噪声和阈值的联合提升能否产生协同效应？
- **C 组（exp08-11）**：只改 Slot Attention 的架构参数（slots 数量和 iters 迭代次数）。回答：聚类质量对防御效果影响多大？
- **D 组（exp12-14）**：改 warmup 时机。回答：CEM 什么时候介入最合适？
- **E 组（exp15-18）**：多维度同时改进，组合 A-D 中表现好的设置。
- **F 组（exp19-20）**：最终候选配置，用来打败 SOTA。

策略是 A-D 做**控制实验**（每次只改一个因素），E-F 做**组合优化**。

---

## 第 13 页 — Group A (exp01-03): CEM Gradient Strength Sweep

A 组只改 Stage 6 和 8 的参数，测 CEM 强度的直接影响。

baseline 的有效强度只有 8.0（lambd=16, loss_scale=0.5）。

- exp01：lambd=24, ls=1.0，有效强度 24.0（3 倍于 baseline）
- exp02：lambd=32, ls=1.0，有效强度 32.0（4 倍）
- exp03：lambd=48, ls=0.8，有效强度 38.4（约 5 倍）

exp03 之所以用 ls=0.8 而不是 1.0，是因为 48×1.0=48 可能太激进，把分类梯度完全淹没，导致分类准确率崩溃。

预期：SSIM 单调下降（防御变好），但 exp03 可能出现分类精度的明显下降。

---

## 第 14 页 — Group B (exp04-07): Noise + Threshold Push

B 组同时推高 noise（Stage 2）和 var_threshold（Stage 6），测试两者的协同效应。

- exp04：noise 从 0.05 提到 0.08，var_threshold 从 0.15 提到 0.20，实际阈值 1.28e-3（3.4 倍于 baseline）
- exp06：noise 只轻微提到 0.06，但 var_threshold 拉到最高 0.25，测试"低噪声 + 严格阈值"策略
- exp07：noise 拉到 0.10（baseline 的 2 倍），是最激进的设置，阈值直接到 2.00e-3（5.3 倍），但代价可能是分类精度大幅下降

---

## 第 15 页 — Group C (exp08-11): Slot Attention Architecture Search

C 组是 Slot Attention 的架构搜索，只改 Stage 3 参数。

- exp08：slots=16（2 倍 baseline），看更多原型是否有帮助
- exp09：slots=4（减半），看极少原型能否工作
- exp10：iters=5（+2 次迭代），看更充分的 slot 竞争是否有益
- exp11：slots=12, iters=4，一个折中方案

slots 的权衡：太少了聚类太粗，特征分散不够精细；太多了可能出现 slot collapse（部分 slot 死掉）。

iters 的权衡：太少可能 slot 竞争不充分；太多增加计算量，而且可能出现梯度通过多层 GRU 迭代后消失。

---

## 第 16 页 — Group D (exp12-14): Warmup Sensitivity

D 组测 CEM 激活时机的敏感性。

- exp12：warmup=1——CEM 几乎从一开始就介入（此时分类准确率还不到 5%）。风险是编码器在还没学会提取有意义特征之前就被迫做分散。潜在好处是特征从一开始就不会"结晶"成容易被攻击的模式。
- exp13：warmup=5——先让分类器充分收敛 5 个 epoch，然后 CEM 再介入。
- exp14：warmup=5, 但 loss_scale=1.2——晚介入的代价用增大 loss_scale 来补偿。

---

## 第 17 页 — Groups E & F: Combined Configurations

E 组和 F 组是综合配置。

E 组（exp15-18）把 A-D 中各方面的改进组合起来，多维度同时调整。

F 组（exp19-20）是最终候选：
- **exp19**：有效强度 40.0，360 个 epoch（最长训练），5 次 slot 迭代。它的特点是给 CEM 全功率提供了最长的时间窗口（约 18 个 epoch）。
- **exp20**：有效强度 43.2（所有实验中最高，baseline 的 5.4 倍），最严格阈值 var_threshold=0.25，最多 slots=12，warmup=5。这是"全力出击"的配置。

---

## 第 18 页 — Summary: Parameter ↔ Stage Mapping

这张总结表列出了所有 9 个可调参数，分别对应哪个阶段、控制什么、增大的好处和风险。

成功标准是：找到一组配置使得**推理 SSIM < 0.532**（打败 baseline），同时分类准确率保持在 60% 以上（合理范围）。

实验正在服务器上跑，结果出来后会更新到 summary.csv 里。

---

## 第 19 页 — Thank You

实验目前正在服务器上运行，20 组实验预计总共需要约 100 小时。结果会自动汇总到 run3log 文件夹下的 summary.csv 中。

谢谢大家，有什么问题可以讨论。

---

## 补充：可能被问到的问题和回答

### Q1: 为什么不直接用更大的 lambd 比如 100？
A: 因为 lambd 控制的是 CEM 梯度注入到编码器的权重。如果太大，分类的 CE 梯度会被完全淹没，模型只学防御不学分类，分类精度会崩溃到接近随机（0.2%）。我们的策略是逐步提升（从 24 到 48），找到准确率和防御的平衡点。

### Q2: Slot Attention 比 KMeans 好在哪里？
A: 三个关键优势：
1. **可微分**：梯度可以直接从 CEM loss 流回编码器，不需要双反向传播中的梯度保存-恢复操作
2. **实时性**：每个 mini-batch 实时计算，不像 KMeans 需要用上一个 epoch 的聚类结果（滞后一个 epoch）
3. **软分配**：每个样本可以属于多个 cluster（软概率），比 KMeans 的硬分配更灵活

### Q3: noise 参数和 CEM 有什么关系？
A: noise 有双重作用：(1) 在 Stage 2 直接给 smashed data 加噪声保护传输通道；(2) 在 Stage 6 通过公式 threshold = var_threshold × noise² 间接抬高 CEM 的方差门槛。所以 noise 同时增强了两层防御。

### Q4: 双反向传播为什么要那么复杂？
A: 因为 CEM loss 和 CE loss 对编码器的优化方向是矛盾的——CE 想让同类特征聚拢（便于分类），CEM 想让同类特征分散（防御攻击）。如果直接加在一起做一次 backward，两个梯度会互相抵消。双反向传播允许我们用 lambd 精确控制两个梯度的相对强度。

### Q5: 自适应 CEM 缩放 s(lr) 是什么意思？
A: 用 CosineAnnealing 学习率调度，学习率从 0.035 逐渐降到接近 0。当 lr > 0.00041（训练前 95% 的时间），CEM 梯度被缩放为 0.001/lr ≈ 0.03 倍（很弱）。只有当 lr < 0.00041 时（最后 5%），CEM 达到全功率。这样做是让模型先学好分类特征，最后阶段才全力做防御微调。
