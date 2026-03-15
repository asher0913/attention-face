# `run2.sh` 实验设置讲稿

这份讲稿用于对应以下幻灯片：

- [run2_experiment_setup_slides.pdf](/Users/asher/Documents/attention-face/run2exp_slides/run2_experiment_setup_slides.pdf)

建议使用方式：

- 每一页 PPT 对应阅读本讲稿中的一节
- 讲的时候不需要逐字逐句全部念完，可以优先念“可直接讲”的部分
- 如果导师追问，再补充“展开解释”部分

---

## Slide 1: Title Page

### 这一页我要讲什么

这一页主要是开场，告诉导师这次汇报的核心不是实验结果，而是解释我的实验脚本 `run2.sh` 是怎么设计的，以及为什么要这样设计。

### 可直接讲

各位老师好，这次我主要汇报的是我在根目录项目里为新架构设计的一个系统化实验脚本，也就是 `run2.sh`。  
这份脚本的目标不是简单地重复原来的实验，而是围绕我现在的新方法，也就是基于 Slot Attention 的隐私保护架构，做一轮更有针对性的超参数搜索。  
因为目前全部实验结果还没有完全跑出来，所以这次汇报的重点不是具体数值，而是实验设计本身：  
也就是这个算法结构里到底有哪些关键步骤，每个步骤受哪些参数影响，我改了哪些参数，为什么这样改，以及我预期这些改动会带来什么效果。

### 过渡句

下一页我先从整体上讲清楚，这个脚本到底在做什么，它和以前那种单点实验有什么不同。

---

## Slide 2: Pipeline Overview and Search Logic

### 这一页我要讲什么

这一页要先建立一个总框架：  
这不是随机试 20 次，而是在一个固定公平骨架上，对新架构最关键的参数做定向搜索。

### 可直接讲

这一页我先讲整个 `run2.sh` 的设计逻辑。  
这次实验的总目标，是在 FaceScrub 数据集上，在一个和原始 CEM baseline 尽量公平一致的 split learning 框架下，找到一组更适合我这个新 Slot-Attention 架构的超参数，让它在 utility 和 privacy 的平衡上有机会优于原始方法。  

这里我刻意把参数分成两类。  
第一类是固定不动的部分，这些是为了公平比较而锁定的。  
比如数据集固定成 FaceScrub，输入固定 64x64，主干固定为 `vgg11_bn_sgm`，切层固定为 `cutlayer=4`，瓶颈固定为 `noRELU_C8S1`，结构性约束固定为 `SCA_new`，攻击协议固定为 MIA，攻击 epoch 固定为 50。  
这些内容我没有放进搜索空间，因为如果这些都一起改，那最后即使结果更好，也很难说明到底是因为我的新 regularizer 更好，还是因为 backbone、attack budget 或其他设置变了。

第二类才是我真正搜索的参数。  
这部分主要围绕新架构本身的几个关键问题展开。  
第一，是 slot 结构要多细，也就是用多少个 slot、做多少轮 refinement。  
第二，是 cross attention 的上下文交互要多强，也就是 heads 的设置。  
第三，是隐私 regularization 要压多狠，也就是 noise、lambda、variance threshold、attention loss scale 这些。  
第四，是训练过程如何稳定下来，也就是 batch size、learning rate、epoch 和 warmup 的安排。

所以这 20 组实验不是 20 次独立乱试，而是围绕三个核心目标做的系统化搜索：  
第一，提升结构表达质量；  
第二，提升隐私强度；  
第三，保证优化过程稳定。

### 展开解释

如果导师问“为什么不是用网格搜索或者随机搜索”，你可以补充：

- 这个问题不是普通分类任务
- 参数之间有明显耦合关系
- 比如 `lambda`、`noise`、`var_threshold`、`attention_loss_scale` 不适合彼此独立乱搜
- 所以我用的是“手工设计的结构化搜索空间”，而不是纯机械网格搜索

### 过渡句

有了整体框架以后，下面我就按照模型真正运行的顺序，一步一步讲整个架构流程，以及每一步受哪些参数影响。

---

## Slide 3: Step 1 - Data Loading and Input Geometry

### 这一页我要讲什么

解释为什么输入分辨率和 batch size 是第一步的重要参数，尤其是为什么 FaceScrub 要固定成 64x64。

### 可直接讲

这一步是整个流程的起点，也就是数据加载和输入几何结构的设定。  
在我的实验里，FaceScrub 图像会统一 resize 到 64x64。  
这个决定不是随便定的，而是直接服务于后面的 Slot Attention 机制。  

原因是这样的：  
我这个新架构不是把 smashed feature 直接看成一个整体向量，而是要把它保留成一个具有空间结构的特征图，再进一步拆成一组 spatial tokens。  
如果输入分辨率太低，比如像 CIFAR 那样 32x32，那么经过前面几层卷积和池化以后，空间结构会被压得太厉害，后面的 slot 机制就很难再从中分辨出稳定的人脸局部结构。  
而 FaceScrub 本身是高保真人脸数据，五官、轮廓、发型、局部纹理都比 CIFAR 那种小图复杂得多，所以 64x64 是一个比较合理的折中点：  
一方面保留了足够的人脸空间结构；  
另一方面计算量仍然是可控的。

这一页里另外一个我搜索的参数是 batch size。  
我让 batch size 在 224、256、288 之间变化。  
这样设的原因是，我这里的 regularization 不是单样本的，它依赖 batch 内部的统计稳定性。  
无论是类条件统计，还是 slot-level variance，batch 太小时都会抖得很厉害。  
所以大 batch 在这个问题里不仅仅是为了加速训练，更重要的是让 robust loss 的估计更稳定。

### 展开解释

可以补充一句：

- 在这个任务里，大 batch 的收益主要体现在统计稳定性，而不只是普通分类里的梯度平均效果

### 预期效果

我预期这一步的设置会带来两个直接好处：  
第一，后面的 token 化和 slot 分解会更有意义；  
第二，batch 内构造出来的 privacy regularization 会更稳定，不容易在不同 iteration 之间剧烈抖动。

### 过渡句

数据进入模型之后，下一步就是客户端编码器和 split point，也就是这个中间表示到底从哪里切开。

---

## Slide 4: Step 2 - Client Encoder and Split Boundary

### 这一页我要讲什么

解释为什么 backbone、cutlayer 和单客户端设置在这轮实验里被固定下来。

### 可直接讲

第二步是 client encoder 和 split boundary，也就是整个 split learning 架构里最基础的骨架。  
在这轮实验中，我把 backbone 固定为 `vgg11_bn_sgm`，把切层固定在 `cutlayer=4`，并且采用单客户端设置。  

这里我特意没有去搜索 backbone，也没有去改 cutlayer。  
原因很明确：这次我想验证的是新 regularizer，也就是新的 slot-attention 架构，到底能不能在相同骨架下优于原始 CEM baseline。  
如果我连 backbone 或 split 点都一起改，那最后结果即使更好，也无法清楚归因。  

为什么 `cutlayer=4` 是合理的？  
因为这个位置是一个比较好的折中点。  
如果切得太浅，那么客户端输出的表示还很接近原始图像，隐私风险会更大；  
如果切得太深，那么中间表示已经太抽象，虽然隐私可能更强，但 slot attention 能利用的空间结构也会减少。  
`cutlayer=4` 处在一个相对中间的位置，既保留了足够的局部结构，又已经不是原始像素级别的低层特征。

单客户端设置的原因也类似。  
这里并不是在研究多客户端联邦环境下的数据异质性，而是先把变量收紧，只比较两件事：  
旧方法的 CEM/GMM 思路和我新方法的 slot-attention 思路，到底谁的 representation regularization 更有效。

### 展开解释

如果导师问“为什么不用更深网络”，你可以补充：

- 这轮实验的目标不是追求分类精度的绝对最优
- 而是做一个和 baseline 可比的 representation-level privacy 对照
- 所以 backbone 变化越少，结论越清晰

### 预期效果

这一步固定下来以后，后面的实验结果就可以更可信地解释为：  
差异主要来自 regularizer 和中间表示建模方式，而不是来自整个网络结构换掉了。

### 过渡句

在确定了从哪里切开以后，下一步就要看这个中间表示以什么形式传出去，也就是 smashed data 的瓶颈设计。

---

## Slide 5: Step 3 - Bottleneck Geometry of Smashed Data

### 这一页我要讲什么

解释为什么固定 `noRELU_C8S1`，以及这个设置为什么对 slot-attention 特别关键。

### 可直接讲

第三步是 smashed data 的瓶颈设计。  
这里我固定的设置是 `noRELU_C8S1`。  
这个字符串其实包含了三个非常关键的决定。  

第一，`C8` 表示把中间表示压到 8 个通道。  
这意味着通信带宽被显著压缩，同时也会迫使模型在一个更紧凑的表示空间里组织信息。  

第二，`S1` 表示 stride 是 1。  
这点在我的新架构里尤其重要。  
因为我后面要做的是把特征图 reshape 成 spatial tokens，再用 slot attention 去建模局部结构。  
如果这里的 stride 大于 1，进一步下采样掉空间尺寸，那么 token 数就会减少，空间布局也会更粗糙，slot 模块能够利用的信息就会变少。  
所以 `S1` 的本质意义，不只是“保留更多特征”，而是“保留后续 slot 分解所依赖的空间网格”。

第三，`noRELU` 表示在这个瓶颈位置不额外加 ReLU。  
我的理解是，这样可以减少在最敏感的中间表示上做额外的非线性截断，从而保留更连续的 feature geometry。  
对于后面的 variance 建模和 token 分解来说，这种连续性通常是有利的。

所以这一页虽然只有一个固定参数，但它其实是整个新方法能成立的前提之一。  
如果这里的空间结构先被破坏了，后面的 slot attention 再强也很难救回来。

### 预期效果

我预期这个设置能带来两个效果：  
第一，保持一个压缩但仍然具有空间语义的中间特征；  
第二，为后面的 tokenization 和 slot decomposition 保留足够的几何结构。

### 过渡句

接下来，在这个瓶颈表示之上，我还保留了原来项目里已有的一层结构化约束，也就是 SCA 部分。

---

## Slide 6: Step 4 - Structured Preconditioning with SCA

### 这一页我要讲什么

解释为什么 `SCA_new` 没有被取消，而是作为共享骨架固定保留。

### 可直接讲

第四步是 SCA，也就是我在项目里沿用的结构化预处理约束。  
在这轮实验中，我把 `AT_regularization` 固定成 `SCA_new`，并把 `AT_regularization_strength` 固定为 0.35，`ssim_threshold` 固定为 0.45。  

这里最关键的一点是：  
我并没有把 SCA 去掉，也没有把它纳入本轮超参数搜索。  
原因是我现在要比较的是在相同 split-learning 骨架下，旧的 CEM/GMM 风格 regularizer 和新的 slot-attention regularizer 谁更好。  
而 SCA 这部分已经属于当前项目共享的结构性约束骨架，如果把它一起大幅修改，那实验的解释会变得很混乱。

换句话说，我这里做的不是“抛弃旧系统、另起炉灶”，而是在一个已有的 split + bottleneck + SCA 框架上，把最核心的表示建模 regularizer，从原来的类分布统计思路，换成新的 slot-based 结构化思路。  

所以 SCA 在这里更像是一层 preconditioning。  
也就是在 slot 模块真正看到 smashed feature 之前，先让这个局部表示不要过于平凡、不要过于直接可逆。  
这样一来，slot 模块接收到的是一个已经有一定结构偏好的中间表示，它更容易进一步学出有意义的人脸部件表示。

### 展开解释

如果导师问“为什么不把 SCA 也一起搜”，你可以补充：

- 因为这轮实验的问题定义是“在固定 shared defense backbone 下，slot regularizer 能否优于 baseline”
- 如果把 SCA 也一起搜，结论就会混成“组合系统调参是否更优”
- 那样不能明确回答新架构本身是否有效

### 预期效果

我预期 SCA 在这里能起到一个底层防线的作用：  
让 smashed feature 本身更难直接恢复，同时又不给 slot 模块制造太大的结构破坏。

### 过渡句

有了这个预处理以后，下一步就进入我新架构真正开始区别于旧方法的地方，也就是把二维特征图转成 token 序列。

---

## Slide 7: Step 5 - Tokenization of Spatial Features

### 这一页我要讲什么

解释新方法和旧方法在表示观上的根本差异：  
旧方法把表示看成一个全局点或分布，新方法把它看成一组 token。

### 可直接讲

第五步是 tokenization，也就是把客户端输出的二维特征图改写成一个空间 token 序列。  
这一步其实是我新方法和旧 baseline 在建模思想上的分水岭。  

原来的方法更倾向于把 smashed data 看成一个整体向量，或者看成一个类条件分布里的一个样本点，然后围绕这个“点”去做 prototype、cluster 或 Gaussian style 的统计建模。  
而我的新方法不这么看。  
我把 smashed data 当成一个仍然保留空间结构的 feature grid，再把这个 grid 拆成一串 token。  

这样做的含义是：  
一张脸不再被理解为“特征空间里的一个整体点”，而是被理解为“由多个局部结构组成的 token 集合”。  
这些 token 之后可以由 slot 来竞争性地解释，从而形成类似“人脸部件”的中间表征。  

这一步没有很多显式搜索参数，但它强烈依赖前面几步的设置。  
比如为什么输入必须是 64x64，为什么 bottleneck 要用 stride 1，为什么 cutlayer 不能太深，本质上都是为了保证到这里时还存在足够有意义的空间网格。  

所以这一页我想强调的是：  
slot attention 能否成功，不只是它自己模块内部的超参数问题，而是整个上游几何设计共同决定的。

### 预期效果

我预期 tokenization 带来的直接收益是：  
后面的 regularizer 不再只关注“这个表示和某个类中心有多远”，而是能关注“这张脸内部有哪些结构区域，这些区域之间如何被分解和约束”。

### 过渡句

有了 token 序列之后，下一步就是用 Slot Attention 去把这些 token 组织成更高层次的部件表示。

---

## Slide 8: Step 6 - Slot Attention: Part Decomposition

### 这一页我要讲什么

解释 `attention_num_slots` 和 `attention_num_iterations` 是如何作用于新架构核心模块的。

### 可直接讲

第六步就是我新方法的核心，也就是 Slot Attention 本身。  
它的作用可以简单理解成：  
给模型一组可学习的 slot，让这些 slot 去竞争性地解释 token 序列，最终把一张脸拆成若干潜在的局部部件表示。  

这一页我搜索的两个关键参数是 `attention_num_slots` 和 `attention_num_iterations`。  

先说 `attention_num_slots`。  
这个参数决定一张脸最终用多少个 slot 去表示。  
我把它从 5 搜到 14。  
这么做的原因是，我不想先验地假设“8 个 slot 一定最优”。  
如果 slot 太少，那么模型可能会把多个局部区域混在一起，比如眼睛、鼻子、轮廓都压到同一个 slot 里，这样结构分解会过粗。  
如果 slot 太多，那么每个 slot 得到的信息量又会太少，容易造成表示碎片化，训练也可能更不稳定。  
所以我把这个范围设成一个中等宽度的人脸部件粒度搜索区间。

再说 `attention_num_iterations`。  
这个参数决定 slot refinement 做几轮。  
Slot Attention 不是一步收敛的，它会重复地做 token 对 slot 的竞争分配，再根据分配结果更新 slot。  
迭代次数太少，slot 还没来得及稳定下来；  
迭代次数太多，计算量会增大，而且在强 regularization 下可能开始过度拟合局部噪声。  
所以我把它设在 3 到 6 之间。  

这两个参数实际上决定的是“结构建模的分辨率”和“结构建模的充分程度”。  
一个控制拆得多细，一个控制每次拆解是否足够稳定。

### 展开解释

如果导师问“为什么 slots 不是越多越好”，你可以回答：

- 因为这里不是一般的多物体分割问题
- 这里的中间表示通道数只有 8，而且任务对象是人脸
- slot 太多会导致每个 slot 承载的信息过少，训练反而不稳定

### 预期效果

我预期这一步的优化会提升两方面：  
第一，让人脸局部结构被更清晰地分解；  
第二，让模型不再把 identity 信息集中压缩进一个非常强、非常可逆的单点表示里。

### 过渡句

但是 slot 自己形成以后还不够，我还让它再回头读取原始 token 序列，也就是下一页的 Cross Attention。

---

## Slide 9: Step 7 - Cross Attention: Slot-to-Token Context Reading

### 这一页我要讲什么

解释 `attention_num_heads` 的作用，以及它为什么只能选 2 或 4。

### 可直接讲

第七步是 Cross Attention。  
它的作用是，在 slot 已经形成初步表示以后，再让这些 slot 回头去看原始 token 序列，从而获得更完整的上下文。  
如果说 Slot Attention 负责“先把局部结构聚起来”，那么 Cross Attention 负责“再让这些局部结构重新看一眼整体布局”。  

这里我搜索的主要参数是 `attention_num_heads`，我只用了 2 和 4 两种设置。  
原因不是我搜索得保守，而是由当前架构本身决定的。  
我这里 slot 和 token 的特征维度实际上只有 8，所以 head 数必须能整除 8。  
理论上 1、2、4、8 都是合法的，但 1 太弱，8 又会让每个 head 只有 1 维，表达力太差，所以真正有意义的选择就是 2 和 4。  

这个参数的作用是控制 Cross Attention 对上下文关系的建模粒度。  
head 少一些，模型更像是在做较粗的全局关系整合；  
head 多一些，它可以从多个子空间去看 token 与 slot 的对应关系。  

我把它和 `num_slots` 配合起来搜索，是因为这两个参数本身就是耦合的。  
slot 数多的时候，如果 heads 也多，模型会更有能力去区分更细的人脸局部关系；  
但与此同时，计算和优化难度也会更高。

### 预期效果

我预期 Cross Attention 调优之后，会让 slot 不只是“局部聚类结果”，而是带有更强全局一致性的结构表示。  
这对分类是有帮助的，同时也可能让 privacy regularizer 更稳定。

### 过渡句

有了 slot 和 cross attention 之后，下一步就进入真正构造隐私 regularization 的阶段，也就是 slot variance 和 Gaussian noise 的联合作用。

---

## Slide 10: Step 8 - Slot Variance Objective and Gaussian Noise

### 这一页我要讲什么

解释这一页是整个 privacy regularization 的核心，包括 `regularization_strength`、`var_threshold`、`attention_loss_scale` 为什么必须联动搜索。

### 可直接讲

第八步是整个新方法里最核心的 privacy regularization 构造。  
在这一阶段，模型会基于 slot 表示去预测每个 slot 的不确定性，也就是方差，然后用这些 slot-level variance 去构造 robustness loss。  
与此同时，模型还会在 smashed data 上加 Gaussian noise。  

这一页我搜索的关键参数有三个：  
第一个是 `regularization_strength`，也就是 noise strength；  
第二个是 `var_threshold`；  
第三个是 `attention_loss_scale`。  

先说 `regularization_strength`。  
它表面上看是在控制 Gaussian noise 的强度，但实际上它不只是加噪的幅度。  
在当前实现里，它还会进入 variance threshold 的尺度计算，也就是说它同时参与了“特征扰动”和“slot variance 目标的标尺”两个过程。  
所以这个参数不能被理解成一个纯粹的 noise knob，它实际上会影响整个 privacy objective 的强弱。

再说 `var_threshold`。  
这个参数决定我希望 slot variance 至少达到多大。  
如果 threshold 低，模型只需要维持比较温和的不确定性即可；  
如果 threshold 高，模型就会被更强地推离低熵、低方差、容易被反演的表示。  
所以它本质上是在控制：  
slot 表示到底允许多确定，还是必须被迫保持一定程度的模糊性和多样性。

最后是 `attention_loss_scale`。  
这个参数控制 slot-based robust loss 自身的原始幅度。  
即使后面还有 `lambda`，这个 scale 仍然有独立意义，因为它决定的是 slot regularizer 本身在 loss 图上的数值量级。  

为什么这三个参数要联动调，而不是分开调？  
因为它们控制的是同一个现象的不同侧面。  
噪声强度增加了，如果 threshold 不动，整个约束的相对尺度就会变；  
threshold 增大了，如果 scale 太小，这个 regularizer 可能仍然不够有影响力；  
scale 很大但 noise 很弱，又可能会让模型在一个不合理的目标上过度优化。  
所以我在 `run2.sh` 里是把它们作为一组强相关参数来设计实验区间的。

### 展开解释

如果导师问“这里的 privacy 强度主要由哪个参数决定”，你可以回答：

- 不是单一参数决定
- 最直接的是 `lambd`
- 但真正决定 slot-based objective 形状的是 `regularization_strength + var_threshold + attention_loss_scale` 这个组合

### 预期效果

我预期这一页的调参会直接影响 privacy-utility tradeoff。  
也就是说：

- 更强的设置通常会让 inversion attack 更难
- 但也更可能损害分类精度
- 所以这里是最典型的 Pareto 权衡区域

### 过渡句

不过有了 robust loss 还不够，关键还在于这个 loss 以什么方式进入优化过程，这就到了下一页的 gradient fusion 和训练日程。

---

## Slide 11: Step 9 - Gradient Fusion and Training Schedule

### 这一页我要讲什么

解释 `lambd`、`learning_rate`、`num_epochs`、`attention_warmup_epochs` 为什么是一组必须联动设计的训练超参。

### 可直接讲

第九步是优化过程本身，也是我认为这个脚本设计里最需要认真讲清楚的一页。  
原因是我这里的训练并不是简单地把分类 loss 和 robustness loss 相加，然后做一次普通的 backward。  
当前代码保留了原项目里的梯度融合策略：  
先对 robustness loss 单独回传，缓存 encoder 梯度；  
再对总损失回传；  
最后再把 robustness gradient 以 `lambda` 加权的形式重新加回 encoder。  

所以在这里，`lambda` 的角色非常关键。  
它不是一个普通 loss coefficient 那么简单，它实际上是在控制 encoder 到底要多大程度上服从 privacy regularizer。  
`lambda` 越大，encoder 越会被推向更隐私友好的表示；  
但同时也越有可能损伤 classification utility。  

接下来是 learning rate。  
因为这里的优化是一个双路径梯度过程，所以 learning rate 的设置比普通训练更敏感。  
尤其当 `lambda`、noise、attention scale 都比较大的时候，如果 learning rate 也过高，训练会很容易不稳定。  
所以我在强 regularization 区间故意把 learning rate 压低。  

然后是 `num_epochs`。  
我把训练轮数从原来更短的设定拉到了 280 到 340。  
这不是单纯为了训练更久，而是因为当前代码里后期的低学习率阶段对 privacy shaping 非常重要。  
越到训练后期，模型越有可能在保持分类能力的同时，把 representation 继续往更鲁棒、更难反演的方向推。  
如果训练太短，这个阶段几乎没有充分展开。

最后是 `attention_warmup_epochs`。  
这个参数控制前几轮是否启用 slot-based regularization。  
我这里把 warmup 设置在 6 到 18，而不是像原来那样很短。  
原因是训练初期 encoder 学到的特征还很混乱，如果太早让 slot loss 介入，它学到的可能不是稳定的人脸局部结构，而只是对早期噪声的一种过拟合。  
所以 warmup 的意义在于：  
先让分类表征基本成型，再让 privacy regularizer 去塑形。

### 展开解释

如果导师问“为什么 lambda 不固定只扫别的参数”，可以这样回答：

- 因为 lambda 决定的是 robust gradient 最终有多强地注入 encoder
- 在当前实现里它是最直接影响 Pareto 前沿位置的参数
- 不扫 lambda，其他很多参数的效果都看不清楚

### 预期效果

我预期这一页的这些训练超参联动以后，会带来三个结果：  
第一，减少训练早期的不稳定；  
第二，增强训练后期的 privacy shaping；  
第三，提高找到 utility/privacy 平衡点的概率。

### 过渡句

前面九步都还是训练内部机制，最后一步就是把每组训练结果都放到同一个攻击协议下进行验证，并完整记录下来。

---

## Slide 12: Step 10 - Attack Evaluation, Logging, and Run Organization

### 这一页我要讲什么

解释为什么脚本不是只训练模型，而是每组都做攻击评估和日志整理。

### 可直接讲

第十步是实验的验证与记录阶段。  
在 `run2.sh` 里，每一组参数都不是只训练一个分类模型就结束，而是必须经过同样的攻击评估流程。  
也就是说，先训练，再调用 `main_test_MIA.py` 去做 MIA 攻击，然后把 utility 和 privacy 两类指标一起记录下来。  

这里我把攻击设置固定为：

- `attack_scheme=MIA`
- `attack_epochs=50`
- `test_gan_AE_type=res_normN8C64`
- 用 best checkpoint 做测试

这些都固定不变，原因非常简单：  
攻击预算必须固定，否则 privacy 指标没有可比性。  
如果某些实验攻击更强、某些实验攻击更弱，那最后根本无法判断是模型更安全，还是攻击不够强。

另外，这个脚本很重要的一点是日志组织。  
每次运行都会新建一个独立目录，把这次 sweep 的 master console log、每组实验的单独日志、每组实验的参数文件，以及最后的 summary CSV 全部存进去。  
也就是说，这个脚本不仅是一个训练脚本，还是一个实验管理脚本。  
它的设计目标是让后续分析变得非常直接：  
我可以在实验全部跑完之后，直接从 CSV 里看每组的 best validation accuracy、train-time MIA 指标和 inference-time MIA 指标，然后再回到单独日志里排查表现好的实验到底发生了什么。

### 展开解释

如果导师问“为什么要做这么完整的日志系统”，你可以补充：

- 因为这是一个 20 组搜索，不是一次性单点实验
- 后续一定需要回看每组训练细节和攻击细节
- 如果不把日志结构化保存，后面很难系统分析 Pareto 前沿

### 预期效果

我预期最后这个设计会让我能够比较清楚地回答两个问题：  
第一，哪一组超参数在分类精度上最好；  
第二，哪一组在不显著损伤精度的前提下，把攻击效果压得最差。  
最终我会从这两个维度去选择候选最优实验，并和 CEM baseline 做正面对比。

### 结束总结可直接讲

最后总结一下，这个 `run2.sh` 的设计逻辑并不是盲目调参。  
它是围绕我这个新架构的 10 个关键处理步骤展开的。  
我固定了 backbone、split point、bottleneck、SCA 和 attack budget，这些是为了保证公平比较；  
我搜索了 batch、lr、epochs、noise、lambda、variance threshold、slots、heads、iterations、attention scale 和 warmup，这些则是因为它们直接控制了新 slot-attention regularizer 的结构建模能力、隐私强度和训练稳定性。  

换句话说，这轮实验真正想回答的问题是：  
在不改变整体公平骨架的前提下，我能否通过更适合 slot-attention 的参数设计，把这个新方法调到比原始 CEM baseline 更好的 utility-privacy 平衡点。  
这就是这份实验脚本的核心目的。

---

## 组会最后可能被问到的问题

### 问题 1：为什么不直接大规模随机搜索？

可以回答：

这个问题里的参数耦合很强，尤其是 `lambda`、`noise`、`var_threshold`、`attention_loss_scale` 和 `warmup`。  
如果完全随机搜，很多实验会落在明显不合理的区域。  
所以我用的是结构化搜索思路：  
先固定公平比较骨架，再对最关键的耦合参数做分阶段搜索。

### 问题 2：为什么不把 backbone 也一起换掉？

可以回答：

因为这轮实验的目标不是证明“任何更大网络都更强”，而是证明在相同 split-learning 骨架下，我的新 regularizer 是否优于原方法。  
如果 backbone 也换掉了，归因会变得不清楚。

### 问题 3：为什么 warmup 要拉长？

可以回答：

因为 slot-based regularizer 依赖中间表示已经具备一定结构性。  
如果在训练很早期就加进去，slot 学到的更可能是噪声，而不是稳定的人脸部件。

### 问题 4：为什么 stride 必须是 1？

可以回答：

因为后续要把 smashed feature 展成 spatial tokens。  
如果 stride 更大，空间结构会过早丢失，slot attention 的优势发挥不出来。

### 问题 5：你怎么判断最后哪组参数最好？

可以回答：

不是只看分类精度，也不是只看攻击指标。  
我会同时看：

- best validation accuracy
- MIA 的 MSE
- MIA 的 SSIM
- MIA 的 PSNR

然后去找 utility 和 privacy 的 Pareto 最优点。

---

## 一句话版收尾

如果时间不够，最后你可以用这段一句话总结：

> 这套 `run2.sh` 实验不是简单扩大训练轮数，而是在固定公平比较骨架的前提下，围绕 slot-attention 新架构的 10 个关键步骤，系统地搜索最影响结构表达、隐私强度和优化稳定性的超参数，目标是找到比原始 CEM baseline 更优的 utility-privacy 平衡点。
