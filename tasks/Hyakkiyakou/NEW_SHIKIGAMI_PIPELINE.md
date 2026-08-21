# 百鬼夜行新式神识别方案

本文记录本仓库当前的新式神识别进度、运行时架构、素材采集流程和扩展方法。目标是让后续维护者或 AI 在不了解历史对话的情况下，也能继续增加新式神并验证修改。

## 1. 当前状态

截至 2026-08-21，独立的新式神识别器支持 21 个目标，共 89 张带透明蒙版的模板，统一阈值为 `0.85`。

| 稀有度 | 中文名 | 运行时标识 | 状态 |
| --- | --- | --- | --- |
| SP | 晨晖惠比寿 | `chenhui_ebisu` | 已支持 |
| SP | 心友犬神 | `xinyou_inugami` | 已支持 |
| SP | 神酿星熊童子 | `shenniang_hoshiguma_dojo` | 已支持 |
| SP | 瑶音紧那罗 | `yaoyin_kinnara` | 已支持 |
| SP | 时曜泷夜叉姬 | `shiyao_takiyashahime` | 已支持 |
| SP | 云间不见岳 | `yunjian_fugengaku` | 已支持 |
| SP | 梦山白藏主 | `mengshan_shiro` | 已支持 |
| SP | 梦引蝴蝶精 | `mengyin_kocho` | 已支持 |
| SP | 蚀月吸血姬 | `shiyue_kyuketsuhime` | 已支持 |
| SP | 天火命铃彦姬 | `tianhuo_suzuhikohime` | 已支持 |
| SSR | 神无月 | `kannazuki` | 已支持 |
| SSR | 思金神 | `shikinshin` | 已支持 |
| SSR | 市加美 | `ichikami` | 已支持 |
| SSR | 龙珏 | `longjue` | 已支持 |
| SSR | 封阳君 | `fengyangjun` | 已支持 |
| SSR | 荒骷髅 | `arakuro` | 已支持 |
| SSR | 卑弥呼 | `himiko` | 已支持 |
| SSR | 雪御前 | `yuki_gozen` | 已支持 |
| SSR | 平将门 | `taira_no_masakado` | 已支持 |
| SSR | 葛叶 | `kuzuha` | 已支持 |
| SSR | 不相狐禅 | `fuso_kozen` | 已支持 |

素材清单中还有 9 个待支持目标：

- SP：龙吟铃鹿御前、遥念烟烟罗、晴思日和坊、灼华桃花妖。
- SSR：祸津神、猫川、鬼金羊、歌留多、毗沙门天。

注意：`new_shikigami_sources.json` 中蚀月吸血姬的素材标识是 `shiyue_ketsukihime`，运行时模板沿用较准确的 `shiyue_kyuketsuhime`。修改时不要误建成两个目标。

## 2. 运行时架构

原 `oashya` 模型继续负责已有式神和 buff。新增式神由轻量的 sidecar 识别器处理，两路结果在进入策略前合并：

```text
1280x720 screenshot
  |-- oashya Tracker -------------------- known shikigami and buffs
  |-- NewRareRecognizer ---------------- curated new SP/SSR templates
                    |
                    +--> merged tracks --> Agent.decision --> click/throw
```

关键文件：

- `agent/new_rare.py`：加载模板并在画面中搜索新式神。
- `new_rare_templates/manifest.json`：阈值、模板文件、来源和原始尺寸。
- `new_rare_templates/*.png`：BGR 图像加 alpha 前景蒙版。
- `script_task.py`：在鬼王选择和撒豆循环中合并两路轨迹。
- `agent/agent.py`：目标选择和 buff 优先级。
- `debugger.py`：连续学习模式下采集完整游戏画面。

`NewRareRecognizer` 的当前行为：

1. 只接受归一化后的 `1280x720` 截图。
2. 搜索纵向范围为 `y=140..620`，图像缩放到 `0.25` 后执行匹配。
3. 使用灰度 `cv2.TM_CCOEFF_NORMED` 和 alpha 蒙版，低纹理误匹配会被方差检查拒绝。
4. 同一式神的多张模板只保留最高分，分数达到 `0.85` 才输出。
5. 输出与 `oashya` 兼容的 track tuple，track id 从 `900000` 开始。
6. 所有 sidecar 目标暂时映射为 `CI.MIN_SP`。策略只需要知道它是 SP/SSR，不依赖真实稀有度分类。

不要把官方立绘直接作为运行时模板。百鬼夜行中的 3D 模型、朝向、缩放和光照与立绘差异很大，必须优先使用百鬼夜行实机帧。

## 3. 当前撒豆策略

`Agent.select_target()` 的优先级为：

```text
(概率 UP 已生效且场上有 SP/SSR -> 减速)
> 概率 UP
> 加速撒豆
> SP/SSR
> 豆子获取
> 好友概率 UP
```

补充规则：

- 场上同时出现 SP/SSR 和概率 UP 时，立即砸概率 UP。
- 没有 SP/SSR 时，概率 UP 移动到 `x <= 720` 才砸。
- 同优先级目标选择最右侧对象，以保留更长的持续撒豆时间。
- 冻结 buff 不使用。此前冻结画面会降低式神识别稳定性，已改为满足条件时优先砸减速。
- 其他 buff、SR/R/N/G 式神默认不砸。

## 4. 采集实机素材

使用项目 Python 环境：

```bash
conda activate oas
```

在对应账号配置的 `Hyakkiyakou.debug_config` 中开启：

```json
{
  "continuous_learning": true
}
```

连续学习模式会保存三次鬼王候选检测画面，并在撒豆阶段每 `0.5` 秒将一张完整游戏画面直接写入磁盘。JPEG 质量为 92；即使任务中途停止，已经采集的帧也不会丢失。文件写入：

```text
log/hya/YYYYMMDDTHH/hya_<timestamp>.jpg
```

采集目录和分析数据位于 `.gitignore` 中，不提交原始截图、视频和中间候选。结束采集后应关闭 `continuous_learning`，避免持续占用磁盘。

## 5. 从截图发现候选

先让原模型离线处理截图并记录轨迹：

```bash
python tasks/Hyakkiyakou/prepare/new_shikigami_candidates.py extract --source log/hya
python tasks/Hyakkiyakou/prepare/new_shikigami_candidates.py status --source log/hya
```

然后从指定采集时段中提取没有被原模型覆盖的前景连通区域：

```bash
python tasks/Hyakkiyakou/prepare/discover_unrecognized.py \
  --capture-dir log/hya/20260816T17 \
  --output-dir tasks/Hyakkiyakou/dataset/new_shikigami/real_hyakki/unrecognized_review \
  --limit 300
```

输出包括：

- `sheet_*.jpg`：每页 20 个候选的人工审查图。
- `candidate_*.jpg`：单个候选裁剪。
- `index.json`：候选编号到原截图和 bbox 的映射。

候选很多不代表存在很多新式神。连通区域算法也会收集撒豆爆炸特效、buff、遮挡、旧式神、皮肤和同一对象的重复帧。

确认候选时必须：

1. 从 `index.json` 找到原截图。
2. 查看时间戳前后的连续帧，确认是同一对象从右向左移动。
3. 与官方展示素材或可靠角色资料核对外形。
4. 排除已支持模板和旧式神皮肤。

## 6. 官方素材辅助检索

目标和素材来源记录在 `prepare/new_shikigami_sources.json`。以下命令用于下载、抽帧和生成总览：

```bash
python tasks/Hyakkiyakou/prepare/new_shikigami_materials.py status
python tasks/Hyakkiyakou/prepare/new_shikigami_materials.py download --name fengyangjun
python tasks/Hyakkiyakou/prepare/new_shikigami_materials.py extract --name fengyangjun
python tasks/Hyakkiyakou/prepare/new_shikigami_materials.py overview
```

`download` 需要 `yt-dlp`。部分站点可能要求浏览器 cookie：

```bash
python tasks/Hyakkiyakou/prepare/new_shikigami_materials.py download \
  --name fengyangjun --cookies-from-browser chrome
```

可选的 CLIP 排序工具用于把候选与官方素材做粗排：

```bash
pip install open-clip-torch
python tasks/Hyakkiyakou/prepare/match_unrecognized_materials.py \
  --candidate-dir tasks/Hyakkiyakou/dataset/new_shikigami/real_hyakki/unrecognized_review
```

CLIP 分数不能作为最终身份判断。官方视频和百鬼夜行画面的域差异很大，曾出现同一旧式神被多个新式神类别同时排到首位的情况。

## 7. 增加一个运行时模板

模板生成脚本是 `prepare/build_new_rare_templates.py`。它使用硬编码的来源目录和人工 bbox，保证每张模板都能追溯到原帧。

步骤：

1. 在 `SOURCES` 中登记实机采集目录或公共视频抽帧目录。
2. 在 `ANNOTATIONS` 中以运行时标识增加 3 到 5 张连续、无遮挡、不同横向位置的帧。
3. bbox 只包住目标式神，尽量避开相邻式神、玩家、buff 标签和顶部 UI。
4. 公共视频背景复杂时使用 `include` 限制允许进入 alpha 蒙版的内部区域。
5. 运行生成器并检查输出 PNG 的 alpha 是否只包含目标。

```bash
python tasks/Hyakkiyakou/prepare/build_new_rare_templates.py
```

生成器依赖本地忽略目录中的原始素材。在干净 clone 中，已提交的 PNG 和 manifest 可以直接运行，但缺少对应原帧时不能重新生成全部模板。扩展模板前先恢复或重新采集脚本里引用的来源。

## 8. 验证要求

每次增加或调整模板至少完成以下验证：

```bash
python -m unittest tests.test_hyakkiyakou_new_rare tests.test_hyakkiyakou_agent
python -m py_compile \
  tasks/Hyakkiyakou/agent/new_rare.py \
  tasks/Hyakkiyakou/prepare/*.py
git diff --check
```

还要用真实采集帧做离线检查：

- 所有人工标注的正样本都应超过阈值。
- 间隔抽样整批截图，检查是否命中不相邻的错误时间段。
- 同一角色的命中应形成连续轨迹，不应在全局随机散落。
- 新模板最低分如果只略高于阈值，应继续补更干净的实机帧，而不是立即降低全局阈值。

2026-08-16 的验证记录：从两个采集时段共 1724 张截图中抽查 105 张，并强制包含所有新增正样本。平将门 4 帧、神无月 5 帧、神酿星熊童子 3 帧、封阳君 3 帧、心友犬神 3 帧全部命中，没有额外误报；最低分为 `0.8959`。

同日 `20260816T22` 的 230 张三局采集帧中确认云间不见岳形成两段连续轨迹，加入 3 张实机模板。人工复核 300 个未识别前景候选后，没有确认到其他待支持新式神。新模板在本批中连续命中 4 帧，分数为 `0.8841..0.9862`；扫描此前三个时段共 2402 张历史截图，误报为 0。

2026-08-19 的 `20260819T03` 实机采集中确认市加美形成连续轨迹，使用 5 张不同横向位置的清晰帧制作模板。她在旧模型中会被识别成低稀有度式神或完全漏检，因此由 sidecar 识别器补充支持。回扫历史采集共 4724 帧，只命中这 5 张连续正样本，分数为 `0.9255..0.9651`，未发现额外误报。

2026-08-21 对 `20260821T03`、`T04`、`T05` 的候选拼图进行人工复核，确认了不相狐禅、卑弥呼、梦山白藏主、天火命铃彦姬和雪御前的实机轨迹。模板分别取自同一角色横向移动过程中的 2 至 4 张清晰帧；平将门、市加美和瑶音紧那罗也在复核中再次出现，但已由现有模板支持。使用新增模板回扫这三个时段共 17646 张截图，命中不相狐禅 4 张、卑弥呼 4 张、梦山白藏主 2 张、天火命铃彦姬 3 张、雪御前 3 张，全部属于对应的真实连续轨迹，未发现误报。

同批采集中发现心友犬神在当前模拟器尺度下多数帧只有 `0.65..0.83`，原有 3 张模板覆盖不足。补充 4 张当前尺度模板后回扫 6188 帧，共命中 5 张真实正样本，其中包含一张此前 `20260816T22` 的漏检帧；新模板分数为 `0.8870..0.9405`，未发现误报。

2026-08-20 的 `20260820T04` 实机采集中确认葛叶形成一条从画面右侧到左侧的连续轨迹。使用中央到左侧 3 张不同横向位置的清晰帧制作模板，并通过人工 `include` 区域排除相邻式神和底部玩家；右侧入场帧因 Buff 牌或准星与模型重叠而弃用。此前自动素材匹配曾将其误列为思金神候选，最终由玩家按游戏模型确认为葛叶。回扫当天四个时段共 19426 帧，命中两段真实葛叶轨迹、共 7 帧，分数为 `0.8691..0.9963`，未发现误报。

2026-08-20 的 `20260820T23` 实机采集中确认思金神在 `23:16:58..23:17:03` 和 `23:17:38..23:17:39` 出现两段连续轨迹。角色特征为白发、白色毛肩、灰白长袍及胸前红色火焰祭器，并与官方展示素材核对一致。第一段选择 4 张从右到左、遮挡较少的帧制作模板，第二段保留作独立正样本验证。使用最终模板回扫该时段 3517 张截图，只命中 5 张真实思金神帧，分数为 `0.9011..0.9866`，未发现误报；一个容易误认普通长袍式神的最左侧背面模板已弃用。

## 9. 提交边界

应提交：

- 运行时代码、策略代码和测试。
- `new_rare_templates/*.png` 与 `manifest.json`。
- `prepare/*.py`、来源清单和本文档。

不应提交：

- `log/hya/` 中的账号实机截图。
- `tasks/Hyakkiyakou/dataset/` 中的视频、抽帧、候选和 CLIP 输出。
- 账号运行状态、凭证或与本功能无关的任务配置。

后续维护的核心原则是：先从实机连续帧确认身份，再制作 masked template，最后用正样本和跨时段负样本验证。不要为了快速增加支持数量而仅凭单张候选或语义相似度入库。
