# `cem-fixed` 相对于根目录原始版的算法架构重构报告

## 摘要

本文档的目标，是系统性说明 `cem-fixed` 目录下的新实现，相对于仓库根目录原始 attention 版本所发生的算法架构变化、训练流程变化、实验脚本变化，以及这些变化背后的设计动机。本文档**只比较 `cem-fixed` 与根目录原始版**，不讨论 `CEM-main` baseline，也不对外部论文结论进行比较。报告采用接近技术论文的组织方式，重点回答三个问题：第一，根目录原始版究竟在做什么；第二，`cem-fixed` 到底重构了哪些核心环节；第三，这些改动为什么使得新实现更适合作为一个稳定、可扫参、可复现实验的平台。

从总体上看，`cem-fixed` 并不是在原始 attention 版上做局部补丁，而是对“attention regularizer 的定义方式”进行了重新设计。根目录原始版的基本思想，是把客户端中间表示拆解为一组空间 token，然后在单张图像内部用 `SlotAttention + CrossAttention` 学出若干 slot，再对 slot 的方差做约束；而 `cem-fixed` 则转向另一条路径：它把每个样本的中间表示视为一个整体特征向量，在类条件范围内，通过 `FeatureMemoryBank` 收集当前 batch 与历史样本，再利用 `SlotAttention` 对“同类样本集合”做可微分的软聚类，并直接由几何距离定义类内方差。也就是说，`cem-fixed` 的关注对象已经从“图像内部的空间部件分解”切换成了“同类样本之间的软聚类结构建模”。

这种重构带来了几个实质性后果。第一，类标签不再只是后验平均时的一个统计分组，而是在 attention 模块的输入组织阶段就发挥作用。第二，方差不再由一个附加预测头间接给出，而是由 soft assignment 下的加权平方距离直接定义，从而使 loss 的几何意义更清晰。第三，原始版中与最终损失耦合较弱甚至近似无效的 `CrossAttention` 路径被实质性移除，取而代之的是更贴近训练目标的 sample-level soft clustering。第四，实验脚本从原来的单点试跑方式，演化为围绕 `bank_size`、`slot_dim`、`slots`、`iters`、`noise`、`var_threshold` 等关键新参数所设计的系统化超参数扫描框架。

## 一、比较范围与项目背景

本报告中的“根目录原始版”，指的是仓库根目录中的 attention 方案，主要由根目录下的 `model_training_paral_pruning.py` 与 `run_exp1.sh` 构成。该版本已经完成了从 CIFAR 向 FaceScrub 的迁移，也引入了 `SlotAttention`、`CrossAttention`、`SlotCrossAttentionCEM` 等新模块，并通过 `run_exp1.sh` 驱动训练与攻击评估。该版本的重点，在于证明 slot-based attention regularization 这条思路能够接入现有 split learning 框架，并在 FaceScrub 上完整跑通。

本报告中的“`cem-fixed` 版”，指的是 `cem-fixed/` 目录下的完整新分支，包括新的 `model_training_paral_pruning.py`、新的 `run_exp.sh`、与之配套的 `main_MIA.py`、`main_test_MIA.py`、数据处理和攻击评估逻辑。这个版本并不是简单复制根目录原始版后更改几个超参数，而是在不推翻整个 split-learning 训练骨架的前提下，对 attention-CEM 的核心定义进行了重写，并围绕该新定义重构了实验扫描脚本与训练期状态管理方式。

需要强调的是，这两个版本之间有大量共享骨架。二者都仍然使用同一套 split learning 训练框架，仍然基于 `vgg11_bn_sgm`、`cutlayer=4`、`noRELU_C8S1`、`Gaussian_kl` 等机制展开，仍然在训练结束后调用 `main_test_MIA.py` 进行攻击评估。因此，`cem-fixed` 的创新不在于替换了整个系统，而在于对**客户端中间表示的 attention regularizer 如何定义与如何优化**这件事进行了系统重构。

## 二、根目录原始版的算法架构

为了说明 `cem-fixed` 的改动，首先必须清楚根目录原始版究竟在做什么。根目录原始版的训练过程，可以概括为以下链路：首先，输入图像经过客户端编码器 `f`，产生中间表示 `z_private`。在 FaceScrub 的当前设定下，这个张量通常具有四维空间结构，即类似 `[B, C, H, W]` 的形式。接下来，如果启用了 attention-CEM 分支，代码会将 `z_private` 从卷积特征图重排为一组空间 token，即从 `[B, C, H, W]` 变换为 `[B, H*W, C]`。从这个时刻开始，模型不再把 smashed feature 当成一个整体向量，而是把它视为一组局部空间位置 token。

在这一组 token 上，原始版首先执行 `SlotAttention`。该模块的职责，是用固定个数的 slot 去竞争性解释这些空间 token。每个 slot 初始由一个可学习的高斯均值与方差生成，随后经过若干轮基于注意力的 token-to-slot 归属更新，再经过 `GRUCell` 与 MLP 进行 refinement。这个设计的直观动机，是让每个 slot 学出一种“潜在局部部件”或“潜在局部语义”的表示。由于 FaceScrub 属于人脸识别数据集，原始版隐含地希望这些 slot 能够捕捉诸如眼睛、轮廓、头发、表面纹理等局部因素。

在 `SlotAttention` 之后，原始版还额外加入了 `CrossAttention`。这个模块的功能，是让已经形成的 slot 再回过头来读取原始 token 序列，从而用多头注意力的方式补充全局上下文。就架构意图而言，这意味着模型并不满足于只通过一次 slot assignment 得到局部表示，而希望再做一次“slot 查询原始特征”的精化过程。随后，模型通过 `head_mu` 与 `head_logvar` 这两个头，从 refined slots 中预测均值与方差，并由方差构造 `rob_loss`。

然而，这一原始设计虽然概念上比较完整，但它在算法上存在若干关键问题。首先，它的 attention 操作对象是**单张图像内部的空间 token**，而不是**同类样本之间的跨样本结构**。这意味着 attention 分支学到的更像是“图像内部分解”，而不是“类条件分布建模”。其次，最终损失的构造方式是对预测方差施加一个下限约束：当 slot variance 低于阈值时才惩罚，这会鼓励 slot variance 上升。再次，标签虽然被传入了 forward，但真正发挥作用的位置很弱，只是在 `per_sample` 已经算完以后，再按类做一次简单平均；换言之，标签没有真正参与 slot 的形成过程，也没有主导 attention 的输入组织方式。最后，`CrossAttention`、`running_mu`、`running_var` 等模块虽然存在，但它们与最终 loss 的闭环并不强，导致原始版在方法叙事上更像“空间 token 的自监督 slot variance regularizer”，而不是一个严格意义上的类条件 CEM surrogate。

## 三、`cem-fixed` 的总体重构思路

`cem-fixed` 最核心的重构，不是把原始版里的某一个模块调得更稳定，而是重新回答了一个根本问题：**attention 分支到底应该在什么对象上工作，才能更接近 CEM 这类类条件隐私正则的本质？** `cem-fixed` 给出的答案是，attention 不应主要作用于单张图像内部的空间 token，而应作用于同类样本构成的集合。也就是说，新的 attention-CEM 不再优先学习“图像内部部件之间如何分解”，而是优先学习“同一个类别的多个样本如何在特征空间中形成若干软原型”。

基于这一思路，`cem-fixed` 首先放弃了原始版那种把 `z_private` 重排为 `[B, N, C]` token 序列的做法，转而将其直接展平为样本级向量 `[B, D]`。这样一来，每个训练样本对应的是一个整体特征，而不再是一组局部 token。随后，模型并不是立即在 batch 内做无标签 attention，而是先按类别将当前 batch 中的同类样本筛选出来，再从 `FeatureMemoryBank` 中取出该类别在历史 step 中积累的特征，与当前 batch 特征拼接成一个同类样本集合。最后，`SlotAttention` 被应用到这个“同类样本集合”上，用若干 slot 去软聚类这些样本特征，并基于软聚类结果定义类内方差与鲁棒损失。

这一改变使 `cem-fixed` 的 attention 模块在语义上发生了根本转向。原始版更接近“单样本内部结构建模”，而 `cem-fixed` 更接近“类条件样本级 soft clustering”。从这个意义上说，`cem-fixed` 依然保留了 `SlotAttention` 作为可微分软聚类器的角色，但它不再把 slot 当作视觉部件原型，而更像把 slot 当作**同类特征簇的可学习软中心**。这使得整个方法和 split learning 下的类条件表示约束之间的耦合变得更强。

## 四、输入表示粒度的变化：从空间 token 到样本级特征

根目录原始版与 `cem-fixed` 最直观的差异，发生在 attention 分支的第一步输入表示上。原始版在 attention 打开后，会将 `z_private` 从卷积特征图变换成一串空间 token，即每个空间位置对应一个低维特征向量。这个设计保留了空间结构，也让 slot 理论上有机会学到局部部件，但它同时带来了两个问题。第一，token 的通道维很窄，注意力的表达能力受到限制。第二，attention 所处理的是每张图片内部的结构，而不是样本间结构，因此标签信息难以深度嵌入其中。

`cem-fixed` 则直接采用 `z_private.view(B, -1)` 的方式，把每个样本的中间表示展平为一个整体高维向量。这一改动意味着 attention 操作的基本单元，从“某张图片中的某个空间位置”改成了“某个完整样本”。从数学上说，原始版中的每个 attention token 对应于局部位置，而 `cem-fixed` 中每个 attention token 对应于一个整样本特征。这样做牺牲了显式空间结构，但换来了更直接的类条件几何意义：attention 分配不再在“局部 patch”之间进行，而是在“同类样本整体特征”之间进行。

这一变化的直接后果，是 attention 模块不再承担局部视觉部件解析的角色，而承担起“类内样本软聚类器”的角色。对于一个以身份识别为主的 FaceScrub 任务来说，这种改写使 attention 与最终分类目标之间的语义距离更短。因为在身份分类问题中，真正需要被保持或约束的，往往是同一身份样本在中间特征空间中的聚类结构，而不是单张脸内部部件的分解能力。

## 五、从弱标签利用到强类条件组织：`FeatureMemoryBank` 的引入

根目录原始版虽然在 forward 里接受 `labels` 和 `unique_labels`，但标签发挥作用的位置非常有限。原始版的逻辑是先对每个样本单独做 attention，再得到每个样本的 `per_sample` 损失，最后才根据标签把这些损失按类平均。换句话说，标签是在最后一层统计时才出现，而 attention 模块本身并不知道哪些样本属于同一类，也不会利用同类样本之间的几何关系来决定 slot 的组织方式。因此，原始版对标签的使用可以理解为一种**弱 class-conditioning**。

`cem-fixed` 则引入了一个决定性的结构：`FeatureMemoryBank`。这个 memory bank 按类别维护每个类最近若干个投影特征，并在每个训练 step 中用 detached feature 更新自身。这样一来，当前 batch 中某个类别的特征，不再只能与同批次样本发生关系，还可以与此前多个 step 中缓存下来的同类特征一起构成更大的类内样本集合。attention 模块看到的对象不再是孤立的单样本，而是由“当前 batch 同类样本 + 历史同类样本”构成的上下文。

这个改动的意义非常大。首先，它让 attention 模块从输入层面就具有类条件属性，而不是在输出层面才做标签分组。其次，它缓解了纯 batch-wise 统计带来的高噪声问题，因为当前 step 的类内结构不再完全依赖 batch composition。再次，它使得 `SlotAttention` 输出的 slot 不再对应某张图里的局部区域，而更像对应某个类别内部的多个软原型。这样一来，`cem-fixed` 的 attention 分支在结构上更接近“类条件软聚类”，而不是“单样本局部解析”。

从工程角度看，memory bank 的加入也改变了 warmup 的意义。原始版中，attention 模块通常在 warmup 结束后才第一次真正创建并参与训练；而 `cem-fixed` 则在训练初期就创建 `attention_cem`，即使 warmup 阶段尚未启用 `rob_loss`，也会先用投影后的特征去预热 memory bank。这一设计使得当 CEM loss 正式打开时，模型已经积累了一定的类内历史特征，而不是从零开始做软聚类。

## 六、从“预测方差”到“几何方差”：鲁棒损失定义的重构

根目录原始版中的方差，不是直接由样本与聚类中心之间的几何关系定义出来的，而是由 `head_logvar` 从 slot 表示中预测出来的。换言之，原始版更像在做“神经网络预测统计量”：slot attention 先形成一组 slot，然后另一个小网络头负责告诉你这些 slot 的方差应该是多少。这样的做法在表达上是灵活的，但也带来一个问题，即 loss 的几何意义被掩盖了。最终的 `rob_loss` 不再明确对应“样本距离某个中心的离散程度”，而更像是“网络认为它的方差应该是什么”。

`cem-fixed` 则彻底重构了这一步。新版本先对同类样本集合做 `SlotAttention`，得到每个样本对每个 slot 的 soft assignment；然后根据 soft assignment 计算 soft centroid；接着用样本特征与对应 centroid 的加权平方距离，直接定义每个 slot 的几何方差。也就是说，新版的方差不是一个额外预测头给出的结果，而是从“样本 - 软中心”的几何关系直接计算得到的。这种定义方式有两个直接优势。第一，它让 loss 与特征空间结构建立了明确的一一对应关系。第二，它使 attention 模块承担的角色更加单纯：只负责提供 soft assignment，而不再额外负责预测统计量。

更重要的是，`cem-fixed` 在这一步中明确对 centroid 做了 `detach`。这意味着 soft centroid 被当作一种目标几何参考，而不是允许模型通过同时移动中心与样本来投机性降低损失。这样设计的结果是，梯度主要会推动编码器与投影层去真正调整样本分布，而不是让聚类中心和样本一起漂移，从而制造一个看似更小的类内方差。

## 七、损失方向的纠正：从方差下限到方差上限

根目录原始版最关键的算法问题之一，在于 `rob_loss` 的方向与类条件聚类正则的直觉并不一致。原始版的公式本质上是在惩罚“slot variance 小于阈值”的情况，也就是说，它希望 slot variance 至少达到某个水平。用直观语言来说，它更像是在鼓励 slot 表示不要过于确定，不要过度塌缩。这种思想并非完全不合理，因为提高不确定性可能有助于降低可逆性；但问题在于，它与“压缩类内特征分布、控制类内离散度”的聚类正则思路并不一致。

`cem-fixed` 对这一问题进行了根本修正。新版本的 `rob_loss` 以 `ReLU(log(var) - log(threshold))` 为核心，只有当类内几何方差高于阈值时才产生惩罚。也就是说，新的目标是把同类样本围绕各个 soft centroid 的离散程度压到阈值以下，而不是强迫它们维持较大的不确定性。这一变化不是简单的符号翻转，而是意味着方法论从“方差下限正则”切换成了“方差上限正则”。

为什么这一点重要？因为在当前 split-learning 和身份分类的上下文中，中间表示既要承载可分类信息，又要避免泄露过于细粒度的可逆结构。如果把 attention surrogate 设计成过度鼓励方差增大，它可能同时破坏类内判别一致性与表征稳定性，最后表现为准确率下降但隐私收益并不显著。而把 loss 改成类内方差上限控制后，attention regularizer 的优化方向就更清楚了：它鼓励同类样本围绕若干软中心收拢，并以可控的离散度形成类内簇。这种目标比原始版更有机会与身份分类任务本身达成一致。

## 八、`CrossAttention` 的角色变化：从主干模块到实质移除

根目录原始版在方法设计上包含 `CrossAttention`，并在 `SlotAttention` 之后又做了一次 slot 到原始 token 的 cross-attention 更新。从模块结构上看，这似乎是在尝试将 slot 表示与原始 token 之间建立更复杂的双向关系，给人一种“局部部件建模 + 上下文精化”的完整印象。然而，从训练目标角度看，最终用于 `rob_loss` 的核心量仍然是 slot 经过头部预测出来的方差，这使得 `CrossAttention` 虽然存在，却没有形成一个特别强的损失闭环。换句话说，cross-attention 的输出是否真正对最终鲁棒目标起决定作用，在原始版中并不清晰。

`cem-fixed` 对这一点采取了更直接的态度。它虽然保留了 `CrossAttention` 类定义以维持文件结构兼容，但在真正的 `SlotCrossAttentionCEM` 中，已经不再使用 cross-attention 作为核心路径。实验脚本中仍保留了 `attention_num_heads` 参数，主要是为了命令行兼容，而不是因为 heads 仍然是该算法的关键自由度。也就是说，`cem-fixed` 明确承认：对于当前这个重构后的目标，最关键的不是 slot 再去读回原始特征，而是 slot attention 所给出的 soft assignment 及其诱导的类内几何结构。

这一步带来的好处是，模型结构更贴近最终 loss，也更容易分析。原始版里存在“模块看起来很多，但有些模块并未真实构成有效优化闭环”的问题；而 `cem-fixed` 则通过减少无效路径，使得 attention 分支的每个部件都更加直接地服务于 soft clustering 和几何方差约束。

## 九、从低维 token 通道到可控投影空间：`slot_dim` 的引入

原始版的 attention 模块直接作用于空间 token 的通道维，而在当前 FaceScrub + `noRELU_C8S1` 的设置下，这个 token 通道维非常有限。直观地说，模型试图在一个很窄的 token 表征上完成 slot 分配、cross-attention、方差建模等复杂任务，这会自然限制 attention 的表达能力。即使空间 token 数量很多，单个 token 自身的表征容量也很小。

`cem-fixed` 通过引入 `proj_down` 和显式的 `slot_dim` 参数，解决了这一问题。它先将展平后的完整样本特征从高维空间映射到一个中等规模的投影空间，例如 128 维或 256 维，再在这个投影空间中做 `SlotAttention`。这样做具有两层含义。首先，attention 不再被迫直接工作在极窄的 token 通道维上，而能在一个更适合软聚类的连续表示空间中运行。其次，`slot_dim` 成为了一个新的实验自由度，使得研究者可以系统探索 attention surrogate 的容量配置，而不是把 attention 的工作维度锁死在原始特征图通道数上。

这个改动并非只关乎计算便利，它实际上也改变了方法的归因逻辑。原始版里，如果 attention 表现不好，很难判断是思路本身不合适，还是因为 attention 一开始就在过于狭窄的空间里工作。`cem-fixed` 引入可控 `slot_dim` 后，attention surrogate 的容量可以单独调节，从而把“架构能力不足”和“损失定义有问题”这两类原因分离开来。

## 十、训练阶段的生命周期管理：从后置创建到早期初始化与预热

在根目录原始版中，attention 模块通常在 warmup 结束、并且满足 attention 分支启用条件之后，才第一次被实例化并加入优化器参数组。这种方式虽然实现上简单，但存在一个明显问题：warmup 阶段 attention 模块完全处于缺席状态，既没有参与前向，也没有积累任何类内历史信息。因此，当 warmup 结束的那一刻，attention 分支相当于“冷启动”，容易面临初始统计不足与训练不稳定的问题。

`cem-fixed` 则把 attention 模块的生命周期前移。新版本在训练的早期 step 就会创建 `attention_cem`，并将其参数注册进优化器，哪怕当前 epoch 仍然处于 warmup 期间。在 warmup 阶段，虽然 `rob_loss` 还不会正式加入训练，但模型会先用当前样本的投影特征去更新 memory bank。这意味着 warmup 不再只是“纯分类预热”，而变成了“分类预热 + 类内特征记忆预热”。当真正进入 attention-CEM 有效阶段时，模型不是从空白统计开始，而是已经拥有一段历史类内特征作为上下文。

这一改动的价值在于稳定性。对于一个依赖跨样本聚类结构的正则器来说，如果在第一天正式启用时没有任何历史记忆，它产生的 soft clustering 很可能非常噪声；而先预热 memory bank，可以把这种不稳定性显著降低。换句话说，`cem-fixed` 把 warmup 从“推迟 attention”变成了“提前准备 attention 所需的状态”。

## 十一、训练主干与梯度融合：保留骨架，但让 attention 分支更有意义

虽然 `cem-fixed` 在 attention surrogate 本身上做了大量重构，但它并没有推翻原有的双阶段梯度融合训练骨架。新旧两个版本都仍然采用相同的思路：先计算 `rob_loss`，对其做一次 `backward(retain_graph=True)`，缓存编码器与 attention 相关参数的梯度；随后将优化器梯度清零，再对分类总损失做反向传播；最后把 `rob_loss` 所产生的梯度按 `lambd` 权重重新加回编码器参数。这种训练方式的本质，是让分类目标与 CEM surrogate 以一种手工控制的方式共同塑造客户端表征。

因此，`cem-fixed` 的改进不在于修改了训练框架的 outer loop，而在于让 attention 分支产生的梯度更有几何意义、更有类条件含义、也更有可能真正对应“控制类内样本分布”的目标。原始版里，即使双 backward 机制存在，attention 产生的 `rob_loss` 依旧可能因为定义不对齐而给编码器施加一个不够稳定的信号；而 `cem-fixed` 则通过重写 loss 与输入对象，让这套梯度融合机制变得更有价值。

## 十二、实验脚本的重构：从单点试跑到系统化超参数扫描

除了模型本体之外，`cem-fixed` 还对实验脚本进行了明显重构。根目录原始版的 `run_exp1.sh` 更像一个“经过若干手动调整后的试跑脚本”。它固定了大部分超参数，例如 `num_epochs=240`、`regularization_strength=0.05`、`lambd=16`、`attention_num_slots=8`、`attention_num_heads=4`、`attention_num_iterations=3`、`attention_loss_scale=0.5`、`attention_warmup_epochs=3` 等，只围绕极少数配置运行。这样的脚本适合验证模型能否工作，但不适合作为一个完整的架构搜索平台。

`cem-fixed` 的 `run_exp.sh` 则明显采用了“实验矩阵”的思想。脚本首先固定 backbone、数据集、batch size、cutlayer、bottleneck、攻击预算等公共骨架，然后把 sweep 的自由度集中在真正由新架构引入、或者被新架构重新赋予意义的参数上，包括 `lambd`、`noise`、`var_threshold`、`attention_loss_scale`、`attention_num_slots`、`attention_num_iterations`、`attention_bank_size`、`attention_slot_dim`、`attention_warmup_epochs` 与 `num_epochs`。从实验组织上看，脚本并不是简单地做笛卡尔积爆炸，而是按照“CEM strength baseline”“noise + threshold push”“bank size sweep”“slot architecture”“slot_dim sweep”“combined strong configs”“aggressive push”这样的主题分组设计 20 组实验。

这说明 `cem-fixed` 的脚本不是附属品，而是算法重构的一部分。因为一旦 attention surrogate 从空间 token 版改成 sample-level soft clustering 版，原来的关键超参数集合也会随之变化。根目录原始版里，`heads` 和 `token refinement` 看上去是核心自由度；而在 `cem-fixed` 中，真正关键的搜索轴变成了 `bank_size`、`slot_dim`、`slots`、`iters` 与 `noise-threshold-lambd` 的耦合关系。换句话说，实验脚本的重构并不是单纯为了“多跑几组”，而是因为模型的有效自由度已经发生了变化。

## 十三、评估链的一致性修复：攻击输入尺寸与训练尺寸对齐

根目录原始版虽然已经把 FaceScrub 训练主链切到了 64x64，但攻击评估链中仍然残留了旧尺寸设定，使得 FaceScrub 在攻击阶段还沿用 `input_dim=48`。这种不一致会导致一个问题：训练时的 smashed feature 与攻击解码器假定的输入几何尺度并不完全匹配，从而使隐私评估的绝对数值带有额外噪声。对于一条最终要用 `MSE / SSIM / PSNR` 来评价隐私泄露难度的实验链来说，这种尺寸错位会削弱结果解释力。

`cem-fixed` 针对这一点进行了修复，把 FaceScrub 攻击阶段的输入尺寸改到了与训练链一致的 64x64 对应设置。这一变化虽然不直接改变 attention-CEM 的数学形式，但它提高了整个实验闭环的一致性。一个更稳定的注意力正则器，如果配上一条尺寸不匹配的攻击评估链，最终仍然可能得不到可信的比较结果。因此，`cem-fixed` 不仅重构了训练端，也对攻击端做了必要的几何对齐。

## 十四、稳定性策略的变化：从隐式跳过到更明确的异常记录

根目录原始版在 attention 分支中包含 `try/except`，当 attention 计算失败时，代码会打印一条错误信息并把 `rob_loss` 设为 0，然后继续执行训练。这种做法的优点是流程不容易中断，缺点则是实验可能在研究者没有充分意识到的情况下悄悄退化为“只剩分类损失 + 噪声”的版本。对于一个以 attention-CEM 为核心贡献的方法来说，这种静默降级会损害实验可信度。

`cem-fixed` 并没有完全取消这种容错机制，但它显著增强了失败可见性。新版本会统计失败次数与 NaN 次数，在 debug 日志里记录完整 traceback，并在首次失败及之后每隔若干次失败时给出显著 warning，明确写出“CEM defense is not active this step”。也就是说，`cem-fixed` 在异常处理哲学上从“默默跳过”转向了“继续容错，但显式暴露问题”。这并不意味着系统已经完全不存在风险，但至少从实验管理角度看，研究者更容易识别某一组实验是否真的在按预期启用 attention-CEM。

## 十五、`cem-fixed` 的方法定位：从空间结构 attention 到 attention-flavored soft clustering

经过上述分析，可以看到 `cem-fixed` 与根目录原始版之间最深层的差异，不是某一个模块被加上或删掉，而是**方法定位本身发生了转移**。根目录原始版更强调“人脸的空间结构可否通过 slot 来分解”，因此它的语义中心是图像内部结构；而 `cem-fixed` 更强调“同类样本在中间表示空间中能否通过 slot 形成软聚类”，因此它的语义中心是类内分布结构。前者更像是一个视觉结构化表示学习模块，后者则更像一个 attention 化、可微分的类条件聚类 surrogate。

这意味着，`cem-fixed` 不再是原始版那种严格意义上的“face-part attention architecture”。它保留了 `SlotAttention` 这个工具，但让 slot 承担的角色发生了变化。原始版里，slot 可以被解释为单张图像的潜在部件；在 `cem-fixed` 中，slot 更适合被解释为类内样本分布的软原型。这个变化本身并不意味着方法退步，恰恰相反，它解释了为什么 `cem-fixed` 更适合作为一个与整个项目兼容的实现：因为在当前任务设定中，真正需要被约束的核心对象是**类条件样本分布**，而不是单张图像内部的局部部件拓扑。

## 十六、结论

综合来看，`cem-fixed` 相对于根目录原始版的改动可以概括为一句话：它把一个“基于空间 token 的 slot variance regularizer”，重构成了一个“基于类条件样本集合的 SlotAttention soft clustering surrogate”。围绕这一中心重构，`cem-fixed` 完成了输入粒度重写、标签利用增强、memory bank 引入、几何方差定义、loss 方向修正、CrossAttention 实质移除、投影空间参数化、warmup 生命周期前移、攻击评估尺寸对齐，以及实验脚本系统化扫描等一整套改造。

因此，如果从方法论上评价这两个版本的关系，最准确的说法不是“`cem-fixed` 比根目录原始版多了几个超参数”，而是“`cem-fixed` 重新定义了 attention 在这个项目中应该扮演什么角色”。根目录原始版证明了 slot-based 思路可以接入现有 split learning 框架；`cem-fixed` 则进一步把这条思路从一个偏探索性的视觉结构正则器，推进成了一个更明确、更类条件、更几何化、也更适合系统扫参的训练模块。就项目演化脉络而言，这是一种从概念验证走向任务对齐与工程闭环的重构。
