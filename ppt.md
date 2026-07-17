# IDN — 智能交付网络（Intelligence Delivery Network）
## 幻灯片源文件（ppt.md）— 供 LLM 生成幻灯片使用

> **全局说明**
> - 标题:《IDN:智能交付网络 — 像 CDN 分发内容一样,把大模型的能力交付到网络边缘》
> - 受众:计算机领域专家(系统 + 机器学习),熟悉 LoRA、SVD、马氏距离、CDN 概念,无需科普基础知识。
> - 风格:学术报告(paper talk)。**每个设计组件先讲清机制(动机 → 设计 → 公式/算法 → 为什么这样设计),实验结果作为机制的验证证据放在机制之后**,而不是只罗列结果。
> - 篇幅:19 页正片 + 1 页备份。
> - 版式:浅底深字,主色深蓝 #1A4B8C,警示/负结果用琥珀色 #D97706;指标一律等宽字体。
> - 图示生成提示词(Figure prompt)保留英文以便图像模型理解,但提示词中已注明**图内文字标签使用简体中文**,并给出确切中文标签文本。
> - 状态标记:✅ 已实现并有实测数据 · 🔄 进行中 · 🧭 本页为新设计(尚未实现)
> - 本 deck 中所有数字均为仓库实测(result/summary.md),除非标注 [目标] 或 [设计]。

---

## 第 1 页 — 标题页

**版式:** 全幅标题页。项目名超大字号,论点句居中,页脚为团队/日期。背景:第 3 页架构图以 15% 透明度作水印。

**内容:**
- 主标题:**IDN:智能交付网络**
- 副标题:*以 CDN 分发内容的方式,在网络边缘以云端精度提供大模型智能*
- 核心论点(单句,大字):"边缘小模型不需要**变成**大模型;它只需要**针对每个请求进入正确的局部参数状态**——并且知道何时回退云端。"
- 页脚:IDN 项目 · 2026 年 7 月 · 云端教师 Qwen2.5-7B / 边缘基座 Qwen2.5-1.5B · 6 项能力

**Figure prompt(背景水印):**
"Minimalist network diagram: a globe with one large glowing 'origin' data-center node in the center, many small edge nodes at the periphery, thin arcs connecting them; small chip icons at edge nodes; flat vector style, single deep-blue monochrome palette, very light, suitable as a 15% opacity slide background watermark. No text labels."

---

## 第 2 页 — 问题:智能被放在了错误的位置

**版式:** 左右两栏。左 55%:柱状对比图。右 45%:三个论点块 + 底部结论条。

**内容(右栏):**
- **纯云端服务:** 质量高,但有往返时延、带宽成本、隐私暴露、中心 GPU 稀缺;吞吐受源站容量限制。
- **纯边缘小模型:** 快、私密、便宜——但能力差距是实测的(每能力 1,000 条保留样本):GSM8K 精确匹配 7B 领先 **+18.5 分**,MBPP pass@1 领先 **+23.7 分**(六能力宏平均 **0.7135 vs 0.6451**)。
- **必须避开的陷阱:** 让小模型*全局*逼近大模型——受容量限制在信息论上不可行;一切压缩/蒸馏方法只是在"精度–体积"折衷曲线上挑一个点,精度损失是内生的。
- 结论条(琥珀色强调):CDN 在 25 年前解决过同构问题——**不是把源站服务器变小,而是把正确的对象缓存到用户身边,并配上一致性协议。**

**Figure prompt(左栏):**
"Grouped horizontal bar chart, six capability rows with Chinese labels (CNN/DailyMail 摘要 ROUGE-L, CoLA 语法判断 准确率, GSM8K 数学 精确匹配, MBPP 代码 pass@1, SQuAD 抽取问答 精确匹配, SST-2 情感 准确率), two bars per row: '1.5B 边缘模型' (light gray) and '7B 云端模型' (deep blue). Values: CNN/DM 0.173 vs 0.180; CoLA 0.784 vs 0.820; GSM8K 0.693 vs 0.878; MBPP 0.525 vs 0.763; SQuAD 0.776 vs 0.717; SST-2 0.919 vs 0.923. Amber outline + callout arrows on GSM8K and MBPP rows labeled '+18.5 分' and '+23.7 分'. Axis 0 to 1, clean scientific chart style, all text labels in Simplified Chinese."

---

## 第 3 页 — 愿景:智能交付网络的总体架构

**版式:** 上 70% 全幅架构图,下 30% 一张 CDN↔IDN 对照表。

**内容 — 对照表(下方):**

| CDN 概念 | IDN 对应物 |
|---|---|
| 内容对象(文件/分片) | **热点能力包** C_k = ⟨路由边界, 适配器坐标, 校验器, 回退策略, 血缘⟩ |
| 边缘缓存服务器 | 边缘节点:冻结的小基座模型 + 能力包库 |
| 缓存命中 / 未命中 | 请求落入可信局部区域 / 落在所有区域之外 |
| 未命中回源 | 回退到云端教师推理(7B → 500B) |
| 缓存失效与 TTL | 能力包陈旧度监控、影子审计、教师版本失效 |
| 分发经济性(字节数) | 能力包交付 = KB(系数)– MB(残差),而非 GB(整模型) |

**Figure prompt(上方):**
"Technical system architecture diagram, left-to-right flow, ALL text labels in Simplified Chinese. Far left: user icon group labeled '用户请求'. Center-left: large rounded box labeled '边缘节点' containing: (1) small box '路由器:查询嵌入 → PCA → 到 K 个能力中心的马氏距离', (2) box '冻结基座模型 1.5B', (3) a shelf of colored cartridge slots labeled '能力包 C1…C6(LoRA 坐标)', (4) diamond '接受门控:置信度+熵+校验器'. Center-right: dashed arrow labeled '回退(未命中)' to box '云端源站' containing '教师大模型 7B…500B' and '能力包工厂:训练/刷新/失效'. Return arrow from cloud to edge labeled '能力包交付:KB–MB 级增量'. Bottom strip: '监控:漂移 · 影子审计 · 陈旧度'. Flat vector, deep blue + gray, generous whitespace."

---

## 第 4 页 — 形式化目标:局部条件逼近,而非全局模仿(机制 1)

**版式:** 左右两栏。左 50%:被否定的目标 vs 采纳的目标(大号公式)。右 50%:约束系统 + 指标定义。

**内容(左栏)— 目标的重写:**
- 否定(划掉,琥珀色):`F_θ0(x) ≈ F_T(x), ∀x` ——"让小模型处处逼近教师"。这是所有蒸馏/压缩工作的隐含目标,也是它们精度必然受损的根源。
- 采纳(深蓝强调框):`F_{θ0+Δ(x)}(x) ≈ F_T(x), x ∈ ∪_k D_k` ——**按请求选择参数修正 Δ(x)**,只在可信区域 D_k 内声明逼近。
- 设计含义:能力提取 = **局部区域发现 + 局部参数修正 + 选择性路由 + 回退机制**,四者缺一不可,这是与"普通微调"的本质分界。

**内容(右栏)— 系统求解的优化问题:**
```
minimize    E[ Cost(ρ(x), x) ]                    # 路由策略 ρ
subject to  R(F_local | Accept) − R(F_T | Accept) ≤ ε   # 选择性风险
            P(Accept = 1) ≥ γ                           # 覆盖率
            时延、显存 ≤ 边缘预算
```
- 指标词汇表:**选择性风险 SelectiveRisk**(仅在本地服务的请求上度量质量差距)、**覆盖率 Coverage**(边缘命中率)、**回退率**。
- 这就是智能交付的 SLA 语言:*"在接受的请求上提供教师级精度,且接受率 ≥ γ"*——用户购买的是带保证的服务等级,不是"通常还行"的小模型。
- 一句给专家的定位:选择性预测(selective prediction)× 参数高效专化(PEFT)× 内容分发系统,三者的接缝处。

---

## 第 5 页 — 方法总览:五阶段流水线

**版式:** 全幅横向五阶段流水线图,每阶段带状态徽章,下方一行标注各阶段产物。

**内容:**
1. **SPECIALIZE 专化** ✅ — 每能力在冻结 1.5B 上训练一个 LoRA Δ_k;全部从同一 θ0、同一目标模块出发(第 4 阶段可分解性的硬前提)。
2. **ROUTE 路由** ✅ — 仅用查询侧表示:嵌入 → PCA → 每能力高斯 (μ_k, Σ_k) → 马氏距离 → 软权重 r_k。
3. **MIX 朴素混合** ✅ — 权重空间直接加权 Σ r_k·Δ_k;**作为有意设计的失败实验**,量化参数冲突,论证第 4 阶段的必要性。
4. **BASIS 共享基编译** ✅(v1)— 对 K 个 delta 做 SVD 得共享权重基 {B_m};每能力 = 稀疏系数 c_k + 残差 R_k;**在系数空间混合,按请求编译**。
5. **POLICE 治理** 🔄 — 接受门控(置信度/熵/校验器)、回退、能力包生命周期(刷新/新建/退役)、秩预算、异构节点分级。

**Figure prompt:**
"Horizontal pipeline of five rounded rectangles connected by arrows, Chinese labels: '专化 SPECIALIZE', '路由 ROUTE', '朴素混合 MIX(设计性失败)' with small amber flag, '共享基编译 BASIS', '治理 POLICE'. Status badges: green check on first four, orange '进行中' on last. Annotation lane below showing artifacts: 'K 个 LoRA 适配器', '路由空间(PCA + K 个高斯)', '冲突诊断报告', '共享基 + 系数 + 残差', '服务/回退策略 + 生命周期'. Flat vector, deep blue accents, all labels Simplified Chinese."

---

## 第 6 页 — 机制 2:SPECIALIZE — 把"能力"物化为可交付的参数修正

**版式:** 左 45%:参数空间示意图。右 55%:机制设计要点(自上而下四块)。

**内容(右栏)— 设计机制:**
- **训练对象:** 对每个通过验证的区域 D_k,求局部修正
  `Δ_k = argmin_Δ Σ_{x∈D_k} ℓ(F_{θ0+Δ}(x), y*(x)) + λΩ(Δ)`,其中监督 y* 可取金标签(v0)或教师输出 F_T(x)(v1,IDN 原生形态——不需要人工标注,教师自己就是标注器)。
- **为什么小模型不加参数也能变强:** 结构不变、参数点改变。θ0 是全任务折中点;θ0+Δ_k 是 D_k 区域的局部最优点。**提升来自"条件化参数选择",不是容量增加**——这是本方案与"训更大模型"的机制性区别。
- **两条硬约束(为第 8 页的可分解性服务):**
  1. 所有 Δ_k 从**同一 θ0** 初始化,作用于**同一组目标模块**(q,k,v,o,gate,up,down;r=16, α=32)——否则 delta 之间不构成可比坐标;
  2. 每个 Δ_k 连同其路由边界、校验器、回退策略、血缘一起封装为**热点能力包 C_k**——交付单元是包,不是裸适配器。
- **验证证据(实测,每能力 1,000 条):** 6 个专家中 5 个超过 1.5B 基座——CoLA +8.9、SQuAD +12.7、SST-2 +4.8、CNN/DM +2.7 分;其中 CoLA/SQuAD/SST-2 专家甚至超过实测 7B。⚠ 但 GSM8K 专家**崩溃(−18.6 分)**:对推理链任务,金标签 SFT 破坏思维链。**发现:推理型能力抗拒朴素 SFT 专化——瓶颈在专家质量,不在路由。** 🔄 修复:教师推理链蒸馏(v1 监督)+ 秩/轮数扫描。

**Figure prompt(左栏):**
"Parameter-space illustration: a large smooth 2D loss-landscape-like blob labeled '参数空间'. One gray dot at center labeled 'θ0(全任务折中点)'. Six short colored arrows leaving θ0 toward six small colored basins, each basin labeled with a capability name in Chinese (数学 GSM8K, 代码 MBPP, 摘要 CNN/DM, 问答 SQuAD, 语法 CoLA, 情感 SST-2), each arrow labeled 'Δ_k'. One arrow (to GSM8K) drawn in amber with a small crack icon, annotated '朴素 SFT 失败:−18.6 分'. Caption inside figure: '能力增强 = 进入正确的局部参数状态,而非增大模型'. Flat vector, scientific, Chinese labels."

---

## 第 7 页 — 机制 3:ROUTE — 仅查询侧的能力区域识别

**版式:** 左 50%:机制分步(公式栈)。右 50%:路由空间散点图 + 实测框。

**内容(左栏)— 设计机制(四步):**
1. **表示提取(仅查询侧,零教师调用):** `h(x) = ψ(x)` = 边缘基座 1.5B 对 prompt token 的末层隐状态均值池化——路由器复用已加载的模型,无额外网络开销。
2. **PCA 构造路由空间:** `z(x) = P_m^T (h(x) − h̄)`,m ≈ 32–64。低维化既降噪又让协方差可估计。
3. **区域几何建模:** 每能力一个高斯 (μ_k, Σ_k)(收缩正则);马氏距离 `d_k(x) = (z−μ_k)^T Σ_k^{-1} (z−μ_k)` 度量"请求落在能力 k 的分布内多深"——比余弦相似度多了各向异性边界。
4. **软路由权重:** `r_k(x) = softmax(−d_k(x)/T)`,温度 T 控制混合锐度;熵 `H(r)` 作为"路由是否含糊"的信号,供接受门控使用。
- **设计取舍:** 不训练路由网络(样本效率、可解释、毫秒级 numpy 推理);代价是分布偏移下概率未校准——见右侧实测。
- **为什么必须保守:** 交叉能力测试显示选错适配器是灾难性的(CoLA 0.74→0.28,SST-2 0.92→0.09)——**负迁移的严重性本身就是"宁可回退、不可乱路由"这一设计的依据。**

**内容(右栏底部)— 验证证据:**
- 拟合集:路由准确率 **0.9967**(1,200 样本),200 条合成 OOD 探针拒绝率 **1.0000**。
- 混合流(分布偏移):准确率降至 **0.9010**,覆盖率 **0.5890**;主要混淆 = 抽取式问答 ↔ 摘要(SQuAD 166 条仅 69 条路由正确,96 条漏到 CNN/DM)。
- ⚠ 偏移下平均路由置信度仍高达 **0.9744** → 距离导出的概率过度自信。🔄 修复:温度/保形校准 + 中间层激活作第二路由特征 + 熵门控前置。

**Figure prompt(右栏):**
"2D scatter plot styled as a PCA routing space, Chinese labels: six colored point clouds with centroid stars labeled 摘要, 语法, 数学, 代码, 问答, 情感; dashed Mahalanobis ellipses around each. The 问答 (orange) and 摘要 (teal) clouds visibly overlap; a few orange points inside the teal ellipse marked with arrows labeled '路由泄漏'. One gray point far outside all ellipses labeled '域外 → 拒绝'. Clean scientific scatter, Chinese text."

---

## 第 8 页 — 机制 4:为什么不能直接加权合并 LoRA(设计性失败实验)

**版式:** 左 55%:上下两面板机制图(上=朴素混合失败原因,下=共享基思路预告)。右 45%:机制解释 + 实测。

**内容(右栏):**
- **朴素混合(模式 B):** `Δ(x) = Σ_{k∈TopT} r_k(x)·Δ_k`,在**权重空间**直接线性组合。
- **失败的机制根源(不是实现 bug):** 各 Δ_k 独立训练,各自的低秩子空间旋转不一致;同一参数上不同能力的修正方向相反;跨能力参数热点重叠。线性组合在不一致坐标系间做加法 → 相消与干涉。
- **为什么仍要完整地做这个实验:** 它把"参数冲突"从直觉变成可测量(逐模块 cos(Δ_i,Δ_j) 与符号冲突率),**为共享基的必要性提供定量证据**——学术叙事上,这是消融的一环,不是弯路。
- **验证证据(实测,混合流 1,000 条):** 路由加权六 LoRA 混合宏分 **0.5779**,低于 1.5B 基座 0.6224(**−4.5 分**)——不如什么都不做,与机制预测一致。

**Figure prompt(左栏):**
"Two stacked panels with Chinese labels. Top panel titled '朴素权重混合(失败)': inside a parameter-space blob, two arrow-vectors labeled ΔW_A and ΔW_B pointing in conflicting directions, their weighted sum arrow visibly short/canceled, amber annotation '子空间干涉:宏分 −4.5'. Bottom panel titled '思路:先统一坐标系再混合': the same deltas projected onto a shared 4-axis coordinate grid labeled 'B1…B4(共享基)', each delta now shown as a sparse coefficient bar vector, a plus node combining coefficient vectors cleanly, arrow labeled '编译'. Flat vector, amber only in top panel, Chinese text."

---

## 第 9 页 — 机制 5:BASIS — 共享权重基 + 稀疏系数 + 残差(本文核心机制)

**版式:** 左 50%:分解与编译的公式栈(学术风)。右 50%:机制设计决策三条 + 能力包定义框。

**内容(左栏)— 机制,分两半:**
- **离线分解(逐层 l、逐目标模块):**
```
ΔW_{k,l} = B_k A_k → a_{k,l} = vec(ΔW_{k,l})     # 物化并展平
A_l = stack(a_{1,l} … a_{K,l})                    # K×d 堆叠
A_l − Ā_l = U Σ V^T                               # SVD
B_{l,m} = unvec(V[:,m]), m = 1…M_l                # 共享权重基
c_k = argmin_c Σ_l ‖ΔW_{k,l} − Σ_m c_{k,l,m} B_{l,m}‖²_F + λ‖c‖₁   # 稀疏系数
R_{k,l} = ΔW_{k,l} − Σ_m c_{k,l,m} B_{l,m} → 低秩压缩               # 残差
```
- **在线编译(按请求,模式 C):**
```
c(x) = Σ_{k∈TopT} r_k(x)·c_k → TopS 硬稀疏化
Δ(x) = Σ_m c_m(x)·B_m + Σ_{Top-1} r_k(x)·R_k
y    = F_{θ0+Δ(x)}(x)
```

**内容(右栏)— 三条设计决策与理由:**
1. **混合发生在系数空间而非权重空间** —— 共享基给所有能力一个公共坐标系,组合从"权重手术"变为"坐标算术";这是对第 8 页失败机制的直接对症。
2. **残差只取 Top-1** —— 残差没有共享坐标系,多残差混合会重新引入冲突;Top-1 保留能力特异细节又不破坏组合稳定性。
3. **必须通过的往返测试** —— 用 c_k + R_k 重构 Δ_k 须复现单专家精度(已达成:代数重构精确,四个全局方向保留 **0.9199** 的总 delta 能量)。
- **能力包正式定义(强调框):** `C_k = ⟨μ_k, Σ_k, τ_k | c_k, R_k | V_k, Π_k | 血缘⟩`。共享基 {B_m} 每节点只装一次;**新能力的增量交付 = 系数(KB)+ 残差(MB)——这就是 CDN 的分发经济性在参数空间的复刻。**

---

## 第 10 页 — BASIS 的实证解剖:能力几何学(结果作为科学发现)

**版式:** 左 50%:残差能量柱状图。右 50%:解读三条 + 琥珀色坦诚框。

**内容(右栏):**
- 混合流宏分:共享基 **0.5675** vs 朴素混合 0.5779——两者接近;共享基**回收了 GSM8K(+1.8 分)**,但 SQuAD 再损 6.6 分(与第 7 页路由泄漏复合)。
- **真正有趣的科学:残差能量极度不均。** ‖R_k‖² 占比:CoLA/GSM8K/SST-2 ≈ 0%、CNN/DM 0.2%——但 **MBPP 99.98%、SQuAD 99.71%**。四个全局方向几乎完整表达分类/摘要/数学的 delta;**代码生成与抽取问答几乎完全能力特异**。
- 解读:共享基客观上是一台"**能力几何测量仪**"——它量化了哪些能力在参数空间共享结构、哪些不共享。这是发现,不只是压缩技巧;也为"哪些能力适合共享基交付、哪些必须整包交付"给出判据。
- ⚠ 坦诚框:当前物化是功能性实验而非存储优势——基+残差共 2.1 GB,原六个 r16 适配器仅 216 MB。🔄 修复:残差秩截断、因子化存储、扩大适配器库(K≫6 个子能力簇)使均值之上的方向更充分。

**Figure prompt(左栏):**
"Vertical bar chart titled '四方向共享基分解后的残差能量占比(%)', Chinese axis labels. Bars: 语法 CoLA ≈0, 数学 GSM8K ≈0, 情感 SST-2 ≈0, 摘要 CNN/DM 0.2, 问答 SQuAD 99.7, 代码 MBPP 99.98. First four bars deep blue and short, last two amber and nearly full height. Horizontal annotation line labeled '4 个共享方向保留总能量的 92%'. Caption strip: '共享结构:分类·数学·摘要 | 能力特异:代码·抽取问答'. Clean scientific chart, Chinese text."

---

## 第 11 页 — 机制 6:POLICE(一)— 一条请求如何被服务:接受门控与决策阶梯 🔄

**版式:** 左 60%:纵向决策流程图。右 40%:门控公式 + 校验器清单 + 时延预算。

**内容(右栏):**
- **接受门控(三个条件缺一不可):**
```
Accept(x,y) = 1{ max_k r_k(x) ≥ τ      # 落在可信区域
              ∧ H(r(x)) ≤ τ_H          # 路由不含糊
              ∧ V_k(x,y) = 通过 }       # 输出通过校验
```
- **设计理由:** 三信号正交——τ 管"在不在域内",τ_H 管"是否骑在边界上",V_k 管"这次输出本身对不对"。任一失败 → 回退,绝不猜。
- **校验器 V_k 按能力定制且廉价:** GSM8K 答案抽取成功;MBPP 沙箱执行 + 冒烟断言;SQuAD 片段合理性;CoLA/SST-2 标签合法性;CNN/DM 长度/重叠度检查。规则先行,无需额外模型。
- **阈值即运营曲线:** τ, τ_H 在验证流上校准,选定 (覆盖率 γ, 选择性风险 ε) 的工作点——**按部署场景选点,不是全局魔数**。
- 时延预算 [目标]:路由 ≈1 ms(numpy)· 硬路由适配器常驻 ≈0 切换 · 基编译 ≤ 数十 ms 且硬路由可缓存 · 边缘解码约为 7B 的 1/4–1/5 · 回退仅未命中部分支付云端往返。
- **飞轮(强调):** 每次回退都产生一条带教师标注的训练样本 → **未命中就是下一轮能力包刷新的免费监督**——这是 CDN 从未有过的自增强回路。

**Figure prompt(左栏):**
"Vertical decision flowchart, ALL labels Simplified Chinese. Start node '请求 x'. Box '嵌入 + PCA 投影(仅查询侧)'. Box '马氏距离 d_k → 软权重 r_k'. Diamond 'max r_k ≥ τ 且 H(r) ≤ τ_H ?' — 否 → red-outlined box '回退云端教师 F_T' annotated '记录样本,用于能力包刷新'. 是 → Diamond 'Top-1 是否显著占优?' — 是 → box '硬路由:加载常驻适配器 Δ_k*'; 否 → box '基编译:c(x)=Σ r_k c_k → Δ(x)=Σ c_m B_m + Top-1 残差'. Both merge into box '边缘解码 y = F_{θ0+Δ(x)}(x)'. Diamond '校验器 V_k(x,y) 通过?' — 否 → same 回退 box; 是 → green node '边缘返回 y'. Tiny clock icons: '≈1ms' at router, '≈10ms' at compile, '云端往返 + 7B 解码' at fallback. Flat vector flowchart, red only on fallback edges."

---

## 第 12 页 — 机制 7:POLICE(二)— 异构节点:一个能力,一个执行器阶梯 🧭 [设计]

**版式:** 左 55%:四级阶梯图。右 45%:机制描述 + 放置策略。

**内容(右栏):**
- **动机:** IDN 假设节点算力异构(手机 NPU → 家庭 GPU → 基站 GPU → 区域机房)。因此**一个能力包在同一路由边界与校验器之下,携带多档执行器等级**:
  - **等级 0** — 仅系数:复用节点常驻共享基 {B_m},交付 c_k(KB 级)。最弱,但交付近乎免费。
  - **等级 1** — 系数 + 低秩残差 R_k(MB 级)。当前实现的默认边缘档。
  - **等级 2** — 高秩教师蒸馏适配器(数十–数百 MB),挂更大边缘基座(如 3B)。
  - **等级 3** — 云端教师本体(源站,精度天花板)。
- **放置策略(背包问题):** 节点公告 ⟨显存, 算力, 常驻基⟩;部署控制器在内存/时延预算约束下,**最大化该节点观测流量组合上的期望选择性精度**;输入是秩扫描 {4,8,16,32,64} 得到的"质量–秩"边际效用曲线。
- **逐级升级取代一次性回退:** 等级 g 未命中/低置信 → 先试 g+1(可在邻近更强节点),再到源站——**智能的缓存层级(L1/L2/源站)**。
- 秩预算假设(待验证):分类 r≤8 饱和;数学/代码需 r≥32 + 蒸馏——**异构性是按能力的,不只是按节点的**。

**Figure prompt(左栏):**
"Pyramid/ladder diagram with four tiers, Chinese labels, bottom to top: '等级 0:共享基上的系数(KB)— 手机/NPU', '等级 1:+ 低秩残差(MB)— 家庭 GPU · 1.5B 基座', '等级 2:高秩蒸馏适配器(数百 MB)— 基站 GPU · 3B 基座', '等级 3:云端教师 7B–500B(源站)'. Left edge upward arrow labeled '未命中/低置信 → 逐级升级'. Right edge downward arrow labeled '交付成本与时延递减'. Small accuracy dial icon on each tier increasing toward top. Legend box: '同一路由边界 + 同一校验器,多档执行器'. Flat vector, deep blue gradient by tier, Chinese text."

---

## 第 13 页 — 机制 8:POLICE(三)— 能力包生命周期:何时刷新、新建、退役 🧭 [设计]

**版式:** 上 55%:状态机图。下 45%:触发条件表。

**内容(下方)— 触发机制表:**

| 监控信号 | 阈值逻辑 | 动作 |
|---|---|---|
| **影子审计**:抽取 p% 已接受请求异步重放到 F_T 比对 | 一致率 < 证书值 | **刷新**:用近期回退+审计数据重训 Δ_k |
| 校验器失败率(按能力) | 相对基线持续上升 | **刷新**,期间收紧 τ |
| **路由漂移**:min_k d_k(x) 滑动均值、覆盖率趋势 | 持续上升 / 覆盖下滑 | 重拟合路由空间(廉价);持续则**分裂区域** |
| **回退聚簇**:对回退日志嵌入做密度聚类 | 簇频率 ≥ p_min 且教师行为稳定 且探针 LoRA 胜过基座(可提取性三查) | **新建**能力包 C_{K+1};增中心;扩基 |
| 能力包流量占比 | 低于下限持续 N 天 | **退役**(小节点先降级后删除) |
| 教师版本变更 | 血缘不匹配 | **全量失效**(CDN 式版本化清除),分批重训 |

- **设计要点(强调):** 回退日志就是*需求信号*——IDN 从边缘失败的地方发现新热点能力,恰如 CDN 用回源记录预热缓存;"可提取性三查"(频率/教师行为稳定/探针有增益)防止为噪声簇建包。

**Figure prompt(上方):**
"State-machine diagram for capability package lifecycle, Chinese labels. Rounded states: '候选(回退请求聚簇)' → '训练中(探针 LoRA + 可提取性验证)' → '在役(服务中,持证书)' → '陈旧(审计不一致或漂移)' → back to '训练中' via edge '刷新'; '在役' → '退役(低流量/被取代)'. A lightning-bolt edge from a cloud icon labeled '教师升级 → 全量失效' pointing into '陈旧'. Under '在役', three small gauges labeled '影子审计', '校验失败率', '路由漂移'. Flat vector, deep blue states, amber for 陈旧, gray for 退役, Chinese text."

---

## 第 14 页 — 端到端实测记分板:诚实的现状

**版式:** 单张大表 + 下方两个结论框。

**内容 — 混合均衡流(1,000 条,六能力均衡):**

| 系统 | 宏能力分 | 路由准确率 | 覆盖率 | 状态 |
|---|---:|---:|---:|---|
| 1.5B 基座(边缘下限) | 0.6224 | — | — | ✅ |
| 7B 基座(云端上限) | 0.6939 | — | — | ✅ |
| 神谕路由 LoRA(路由上界) | 待出 | 定义为 1.0 | — | 🔄 运行中 |
| 语义路由 + 朴素六 LoRA 混合 | 0.5779 | 0.9010 | 0.5890 | ✅(预期失败) |
| 语义路由 + 四基/Top-1 残差 | 0.5675 | 0.9010 | 0.5890 | ✅ v1 |

- **结论框 1(数字说明什么):** 全链路机制已经打通(路由→编译→服务),基分解代数精确往返,真实路由下四项能力仍有净增益(CoLA +6.6、MBPP +6.0、CNN/DM +2.6、SST-2 +1.8)。综合分暂低于基座,由**两个已定位且可分离的失败**拖累:GSM8K 专家质量、SQuAD 路由泄漏。
- **结论框 2(为什么对架构有信心):** 每个失败都映射到**特定阶段的特定机制修复**——专家侧换教师推理链蒸馏;路由侧做校准与 QA/摘要分离;其余交给"接受门控 + 回退"吸收(架构从未承诺"边缘处处赢",承诺的是**γ 覆盖下的教师级选择性精度**)。待出的神谕 LoRA 结果将干净地把专家失败与路由失败分账——这是下一个决定性对照。

---

## 第 15 页 — 相关工作:各家占了哪块,IDN 占据哪条接缝

**版式:** 2×3 卡片网格,每卡一句"为什么它不是 IDN";底部一条定位横幅。

**内容:**
- **压缩/剪枝**(SparseGPT、Wanda、LLM-Pruner、Sheared-LLaMA、LLM-Sieve):在"精度–体积"曲线上取点;无路由、无选择性保证、无交付语义。
- **适配器组合**(LoRAHub、AdapterFusion、TIES/DARE):合并期的适配器算术;无按查询条件化的路由边界、无风险受控服务、无生命周期。
- **模型级级联/路由**(FrugalGPT 一族):在整模型之间选路;无按请求的参数编译、无共享基分发经济性。
- **投机解码**:输出精确但每个请求都需要大模型在环;是传输层优化,与 IDN 正交(可用于我们的回退路径内部)。
- **语义缓存 / KV 弹夹**(GPTCache;Cartridges):缓存的是知识/答案,不是*参数化能力*;互补——可作等级 0 之下的更廉价档位。
- **选择性预测**:提供风险形式化;从未与参数态路由和能力包交付耦合。
- **定位横幅:** *IDN 的主张在接缝处:按查询条件化的局部参数状态 + 共享坐标系下的能力编译 + 选择性风险 SLA + CDN 级包生命周期——作为一个系统。*

---

## 第 16 页 — 与"普通微调"的机制对照(一页正面回应质疑)

**版式:** 单张两列对照表(学术答辩风格),底部一句话。

| 维度 | 普通微调 | IDN |
|---|---|---|
| 目标 | 全局 X→Y 拟合 | 可信区域内的**选择性逼近**(ε, γ 受约束) |
| 交付对象 | 一个模型/适配器 | **能力包**:区域边界+坐标+校验器+回退+血缘 |
| 区域来源 | 人工数据集标签 | 查询表示聚类 + 教师行为稳定性 + 可提取性验证 |
| 多能力组合 | 训练后合并权重(冲突) | 共享基**系数编译**(坐标算术) |
| 风险语义 | 平均精度 | 选择性风险 + 覆盖率(SLA 语言) |
| 失败处理 | 照答不误 | 不确定/校验失败 → **回退云端** |
| 更新机制 | 无(重训) | 影子审计/漂移触发的**生命周期管理** |
| 系统约束 | 训后再看 | 时延/显存/交付字节数为内生约束 |

- 底部一句话:**微调只是 IDN 五阶段中的一个算子;把它称作"就是微调",相当于把 CDN 称作"就是复制文件"。**

---

## 第 17 页 — 路线图

**版式:** 三泳道横向时间线(当前 ≤1 月 / 近期 1–3 月 / 远期 3–12 月)。

**内容:**
- **当前:** 神谕 LoRA 对照(分离专家失败 vs 路由失败)· GSM8K 专家 v2(教师推理链蒸馏)· 路由校准 + QA/摘要分离特征 · E5 接受门控全流(带 7B 回退)→ 第一条 (覆盖率, 选择性风险) 运营曲线。
- **近期:** 跨能力秩预算背包 · 残差压缩(消掉 2.1 GB → ≤216 MB 的差距)· 异构节点执行器阶梯仿真 · 能力包生命周期仿真(漂移→刷新;回退聚簇→新建)。
- **远期:** 真实流量的无监督区域发现(摆脱基准数据集标签)· 更大适配器库 → 更厚共享基(K≫6)· 多节点放置优化 · 教师规模上探(7B → 72B+ 源站)· IDN 论文:*作为一等网络对象、带 SLA 的能力包*。

**Figure prompt:**
"Three-lane horizontal roadmap timeline, Chinese lane labels: '当前(≤1 个月)', '近期(1–3 个月)', '远期(3–12 个月)'. Milestone chips as rounded rectangles per the content list (Chinese text). A vertical dashed line labeled '首个端到端 SLA 演示(E5)' between lane 1 and 2. Deep blue chips for engineering items, teal for science items, one amber chip labeled 'GSM8K 专家 v2(已知失败修复)'. Minimal flat style."

---

## 第 18 页 — 结语

**版式:** 居中三行 + 小架构缩略图。

**内容:**
- 大字:**"不要交付模型,交付能力。"**
- 支撑句:冻结的边缘基座 + 可路由、可编译、可认证的能力包 + 诚实的回退 = 需要精度的地方给云端精度,能给低时延的地方给边缘时延。
- 讨论题:能力几何(为什么代码与抽取问答拒绝共享基?)· 偏移下的路由校准 · 智能交付的 SLA 形式化 · 源站扩展到 500B。

---

## 第 19 页 — 致谢 / Q&A

**版式:** 极简。项目名 + Q&A 大字 + 仓库/工件索引小字(artifacts/eval · artifacts/routing · artifacts/shared_basis)。

---

## 备份页 B1 — 混合流逐能力完整数据(答辩用)

**版式:** 单张密表 + 答辩注记。

| 能力 | 1.5B 基座 | 7B | 路由 LoRA | 共享基 | 基 vs 基座 |
|---|---:|---:|---:|---:|---:|
| CNN/DM ROUGE-L | 0.2098 | 0.2202 | 0.2376 | 0.2355 | +0.026 |
| CoLA 准确率 | 0.7784 | 0.7784 | 0.8443 | 0.8443 | +0.066 |
| GSM8K 精确匹配 | 0.6527 | 0.8503 | 0.4072 | 0.4251 | −0.228 |
| MBPP pass@1 | 0.5329 | 0.7725 | 0.6048 | 0.5928 | +0.060 |
| SQuAD 精确匹配 | 0.6145 | 0.6024 | 0.4096 | 0.3434 | −0.271 |
| SST-2 准确率 | 0.9458 | 0.9398 | 0.9639 | 0.9639 | +0.018 |

- 答辩注记:SQuAD 损失可分账为路由泄漏(166 条仅 69 条路由正确)+ 共享基截断(再 −6.6);GSM8K 是专家质量问题(第 6 页),共享基反而小幅回收(相对朴素 +1.8)。流中 407/1,000 条与早期评测重叠(评测重叠,非训练泄漏)。路由器拟合集 0.9967 / OOD 拒绝 1.0(200 探针);混合流 0.9010 / 覆盖 0.5890 / 平均置信 0.9744(偏移下过度自信)。
