# PREREG：正式實驗預註冊

> **狀態：FROZEN（第 3 版，使用者 2026-10-31 核可凍結）。** 凍結後，正式實驗（seeds 3000–3039）照本文件跑，所有分析照本文件報告；凍結後的任何改動都算偏離（見 §7）。
> 起草日 2026-10-30。凍結 commit 的 hash 記在 `docs/PROGRESS.md`「PREREG 凍結」一段（hash 無法寫進產生它的 commit 本身）（同一個 commit 含本文件與正式分析腳本 `experiments/prereg_analyze.py`，見 §7、§10）。
> 本文件的數字都附出處（repo 內路徑）。pilot 的結果是已看過的資料，不作為證據（附錄 A）。
> 目標讀者：沒看過對話的人，只讀本文件就能寫出評估腳本（§10）。

## 1. 研究問題與主要假設

**研究問題。** N=3 台 quadrotor 用繩吊同一個 payload。每機只有帶雜訊的 IMU、自己的位姿、繩方向。隊友之間可以分享兩種不同的東西：**隊友幾何**（隊友的機位與繩向，用來推出 payload 姿態與姿態感知的張力分配），以及**估計**（隊友的量測與資訊對，用共識共同估計共享量 θ_s=(δm, δc) 與各機推力係數 ζ_i）。本研究問：

**主線**（使用者決定，`docs/PROGRESS.md`「使用者決定（2026-10-30，拆解後）」）：**分享值不值得、該分享什麼，取決於本地方法的強度與感測品質。**

用語規定：本文件與論文一律分開說「分享隊友幾何」與「分享估計」，不寫籠統的「分享」。

記號：RMSE_eff 見 §4；差值一律 A − B，負 = A 較好；**Δ = 0.5 cm** 是「實質意義」門檻（使用者決定）。s 是感測強度比例（§2）。每個檢定都在**單一軌跡**（circle 或 fig8）內做，以 seed 為單位配對。

### 1.1 假設

| 假設 | 內容 | 檢定（H0 → 備擇） |
|---|---|---|
| **H1a** | 知道隊友位置（幾何）有幫助：Da(s) = RMSE_eff(EKF-L+GEO) − RMSE_eff(EKF-L) < 0 | 單尾 sign-flip；H0: 均值 ≥ 0 |
| **H1b** | 在已知隊友幾何之上，再交換估計有幫助：Db(s) = RMSE_eff(CON-ATT) − RMSE_eff(EKF-L+GEO) < 0 | 單尾 sign-flip；H0: 均值 ≥ 0 |
| **H2** | 簡單本地方法 DOB 的總分享收益隨 s 增加：B_DOB(s) = RMSE_eff(DOB) − RMSE_eff(CON-ATT)，每 seed 對 s 的 OLS 斜率 > 0 | 單尾 sign-flip；H0: 斜率均值 ≤ 0 |
| **H3** | CON-ATT 與 CEN-ATT 在 ±0.25 cm 內等價：D3 = RMSE_eff(CON-ATT) − RMSE_eff(CEN-ATT) | TOST（兩個單尾 sign-flip，p 取較大者）；敏感度分析 ±0.5 cm |
| **H4（次要）** | 兩種收益的相對大小隨 s 改變：X(s) = Da(s) − Db(s)，每 seed 對 s 的 OLS 斜率 < 0（隨 s 增加，幾何收益相對於估計交換收益越來越大） | 單尾 sign-flip；H0: 斜率均值 ≥ 0 |

- **H1a、H1b 在 s ∈ {0, 0.5, 1, 1.75, 2.5} 每個格各報告**，兩條軌跡分開檢定（speed 1.0）。
- **某個 (假設, s) 稱為「成立」** = 兩條軌跡**都**被 Holm 拒絕（交集檢定，§5）。只有一條軌跡被拒絕 → 「單軌成立」（不算成立），並分軌跡描述（真值表見 §1.2）。**不做跨軌跡推論**：兩軌跡都過只表示這兩條軌跡上都有效，不推廣到其他軌跡。
- **Da、Db 不是機制的可加分解。** 它們是一個特定順序（先加幾何、再加估計交換）下的增量；換順序可能不同。論文只寫「在已有 X 的前提下，再加 Y 的增量」。總收益 CON-ATT − EKF-L 只當描述性（報點估計與 CI）。
- **「收益來源隨 s 交叉」**只當描述性結論（直接看 Da(s)、Db(s) 的點估計與 CI）。H4 是唯一的正式檢驗，列為次要假設，只主張斜率的方向，**不**主張「在某個 s 交叉」。
- **H2 的 s 軸揭露：**s 不是單一線性尺度。延遲是階梯（s<0.5 為 0 步、<1.75 為 2 步、其餘 3 步）；`pos_std = 0.01·min(s,1)` 在 s>1 飽和；位置量測有 1 m 截斷（RMSE_eff）；OLS 斜率的 x 軸用 s 的數值（0, 0.5, 1, 1.75, 2.5），所以斜率是「平均每單位 s」的粗摘要，不是結構參數。**DOB（fc=16）與 LOC-M 的 hp 都是在 s=1 調的**，s=0 與高 s 對它們不一定公平（`results/p2d/best.json`）。
- 「點估計是否非遞減」（原 H2 的第二條件）**改為描述性**：只在報告中列出 B_DOB(s) 的五個點估計，不影響結論。

### 1.2 結果的三層判讀（H1a、H1b 每個 s 都套用）

1. **檢定 0（Holm 家族內）**：兩軌跡都被拒絕 → 「成立」；否則 → 「不成立」。「不成立」不等於無效：先看下面的第 3 層。
2. **Δ（實質意義，描述性，不另做檢定）**：在「成立」的格，兩軌跡的點估計都 ≤ −Δ（**恰等於 −0.5 cm 算達標**，即 ≤） → 寫「達實質門檻」；否則寫「**統計上可偵測，但小於事先定的實質門檻**」（0 > 點估計 > −Δ）；兩軌跡分在 Δ 兩側 → 分軌跡寫。
3. **不成立時，完整真值表**（對每個 (假設, s)，依序判斷；程式 `hyp_evidence`）：
   - 兩軌跡都在 Holm 後拒絕 → 「**成立**」。
   - **任一**軌跡的 95% bootstrap CI 完全落在不利方向（H1a/H1b/H4/DROP：CI 下界 > 0；H2：CI 上界 < 0）→ 「**反向**」（前者較差），照實報告，標出是哪條軌跡。
   - 否則，恰有一條軌跡 Holm 後拒絕 → 「**單軌成立**」：分軌跡敘述，**不算主張成立**。
   - 否則（含 CI < 0 但沒過 Holm、CI 包含 0）→ 「**證據不足**」（power 見附錄 B，高 s 的 H1b 預期屬於這一種，不可寫成「無效」）。
   - H3（TOST）不成立 → 「未等價」。**本研究不對 H1a/H1b 做等價檢定**，所以不寫「證實無差異」。

### 1.3 推翻條件（假設不被支持的條件）

| 假設 | 不被支持（任一成立） |
|---|---|
| H1a（在 s） | 該 s 任一軌跡 Holm 後 p ≥ 0.05。若任一軌跡 CI 完全 > 0 → 反向 |
| H1b（在 s） | 同上 |
| H2 | 任一軌跡的斜率檢定 Holm 後 p ≥ 0.05，或斜率均值 ≤ 0。**成立只表示 B_DOB(s) 的平均斜率為正，不代表單調或線性** |
| H3 | 任一軌跡 TOST（Holm 後）p ≥ 0.05，或 D3 的 90% bootstrap CI 超出 (−0.25, 0.25) |
| H4 | 任一軌跡斜率檢定（F5 內 Holm）p ≥ 0.05，或斜率均值 ≥ 0 |

## 2. 實驗格子

固定：N=3、無風、通訊理想（除掉包格）、軌跡 circle、fig8（分開報告）、MPPI 用 `configs/mppi_frozen.json`、`eta_comp` 見 §3。profile 用 `src/smpc/eval/sensing.py` 的 `get_profile`，字串 `"s=<值>"`；s=0 即 clean、s=1 即 moderate（程式保證逐位相同）。

| 格 | 設定 | seeds | 方法 |
|---|---|---|---|
| **G1 主格** | s ∈ {0, 0.5, 1, 1.75, 2.5}，speed 1.0 | 3000–3039（40） | DOB、LOC-M、EKF-L、EKF-L+GEO、CEN-ATT、CON-ATT |
| **G2 速度（次要）** | s=1，speed 1.5 | 3000–3039 | 同 G1 的 6 個方法（hp 是在 speed 1.0 下調的） |
| **G4 掉包（小規模）** | s=1，speed 1.0，CON-ATT 的 p_drop=0.5（p=0 沿用 G1）；p=0.9 描述性 | p=0.5：3000–3039；p=0.9：3000–3019 | CON-ATT（`hp["p_drop"]`，`iters=2`） |
| **G5 描述** | s=1，speed 1.0 與 1.5 | 3000–3039 | HYB（speed 1.0、1.5）；ORACLE（僅 speed 1.0） |

（編號沿用舊版：舊 G3 台數 sweep 已移出，見 §8。）

- 感測各 s 的數值由 `scaled_profile` 決定：IMU σ、偏差、kerr、繩方向 σ 從 clean 到 moderate 線性外插；延遲 s<0.5 為 0 步、<1.75 為 2 步、其餘 3 步；`pos_std = 0.01·min(s,1)`（`src/smpc/eval/sensing.py`）。
- **掉包與 seeds：**p=0.5 與 p=0 用**同一批 seeds**（3000–3039）。seed 決定真值 θ、k_true、感測雜訊流、初始條件與 MPPI 抽樣流；掉包的亂數流另由 `[seed, 6]` 獨立產生（`con_att.py`），所以 p=0.5 與 p=0 的差只來自掉包，可按 seed 配對。p=0.9 用其中前 20 個 seeds（3000–3019），描述性。
- p=0.9 列為**描述性**：pilot 顯示 p=0.9 時 ζ／δm 誤差與 clip 比例反而變差、RMSE 卻變好，推論為 HCMCI 在高掉包下過度自信（`results/p2d/P2d_report.md` §4）。不做假設檢定，不引用為「掉包有益」。EKF-L+GEO 沒有實作掉包。
- seeds 2020–2029、2100–2119、2200–2219 是已看過的資料，不進主表。

## 3. 方法、可用資訊與凍結 hp

### 3.1 方法 × 可用資訊

直接採用 `src/smpc/adapt/ekf_geo.py` 檔頭的表，並對照 `episode_p2c.py`、`local_baselines.py`、`ekf_local.py`、`con_att.py` 核對過（✓ = 該機的估計器會讀；✗ = 不讀）：

| 方法 | IMU(y,f) | 自身位姿 | 繩向 | 隊友位置(pq_j, q_j) | 隊友估計 | 隊友量測(y_j, f_j, active_j) |
|---|---|---|---|---|---|---|
| DOB | ✓(自己) | ✗ | ✗ | ✗ | ✗ | ✗ |
| LOC-M | ✓(自己) | ✓ | ✓ | ✗ | ✗ | ✗ |
| EKF-L | ✓(自己) | ✓ | ✓ | ✗ | ✗ | ✗ |
| EKF-L+GEO | ✓(自己) | ✓ | ✓ | ✓(轉傳 N−1 輪) | ✗ | ✗ |
| CEN-ATT | ✓(全隊) | ✓ | ✓ | ✓(集中) | (集中) | ✓(集中) |
| CON-ATT | ✓(自己) | ✓ | ✓ | ✓(轉傳 N−1 輪) | ✓(資訊對 Λ,ξ，共識) | ✓(轉傳 N−1 輪) |

註：
- (a) 「隊友位置」＝隊友機位 pq_j **與其繩向 q_j**（合成掛點 pa_j = pq_j + L q_j）。姿態感知的張力分配需要全隊繩向，所以 GEO 與 CON-ATT 的封包都含 q_j（每個來源共 6 個數）。所以 H1a 的「幾何」包含隊友繩向，不只是機位。
- (b) DOB 只用 y 的低通；其他方法的「自身位姿」欄，指由自身位置與參考算出的 slack 旗標 `active`，以及自身繩向 q。
- (c) 所有方法都不讀 payload 真值。MPPI 控制器對所有方法相同（同一組 `mppi_frozen.json`）。
- (d) 結論用語：EKF-L+GEO − EKF-L 的差 =「分享隊友幾何」的增量；CON-ATT − EKF-L+GEO 的差 =「分享估計」的增量（在已有隊友幾何之上）。EKF-L+GEO 不做力矩偽量測（那需要 y_j）。

### 3.2 凍結 hp 與補償

hp 凍結為 `results/p2d/best.json`；正式實驗不再調參，所有格共用同一組（含所有 s、軌跡、speed；speed 1.5 不另調）。

| 方法 | 角色 | 凍結 hp（出處 best.json） | eta_comp |
|---|---|---|---|
| DOB | 簡單本地基線（不通訊、無模型前饋） | fc=16 | none |
| LOC-M | 簡單本地（LocalDm 模型前饋＋ζ̂） | lam=0.998, prior=1 | est |
| EKF-L | 強本地（狀態 [δm, ζ, IMU 偏差 b]） | q_b=5e-4, r_scale=10 | est |
| EKF-L+GEO | 本地 EKF 加隊友幾何（拆解對照，H1a/H1b 的中介） | q_b=5e-4, r_scale=10, use_dc=0 | est |
| CEN-ATT | 集中式，**參考上界**，用於 H3 | lam=0.9998, prior=2 | est |
| CON-ATT | **主要分享估計的方法**（path 圖 HCMCI，每步 2 次迭代，鄰居交換） | 同 CEN-ATT；`iters=2`，`p_drop`=0（G4 另設） | est |
| HYB | **描述性**：CEN-ATT 模型預測＋DOB 殘差 | fc、tau 取 `results/p2c/best.json` 的 `HYB|moderate|{1.0,1.5}`（speed 1.0：fc=8, tau=0.07；speed 1.5：fc=1000, tau=0.07） | est（`y_hyb` 修正後路徑） |
| ORACLE | **描述性**：真參數前饋（無回授），不當上界 | 無 | oracle（真 k_true） |

- 規則：有模型前饋的方法一律用 est 補償（使用者決定）。DOB 無模型前饋，故為 none；ORACLE 用真值前饋，故用真 k_true，不屬於 est。實作對應 `experiments/p2d_run.py` 的 `EST`/`eta_of`。
- **HYB 與 ORACLE 只描述，不進任何 Holm 家族。**HYB 的 hp 是在舊 P2c 協定（moderate/harsh，尚未用 est 補償）下調的，**不是在補償條件下調的**，所以 HYB 不寫成「比 CEN-ATT 更好／更差的新方法」。ORACLE 用 oracle 補償，不當成績效上界（`docs/PROGRESS.md`「P2c 紅方第 1 輪裁決」S1、M3）。

### 3.3 調參預算表

調參條件：moderate×1.0、seeds 2020–2024（5）、circle+fig8（每個 hp 點 10 集）；選擇指標為平均 RMSE（>1 m 或發散記 1 m）。來源 `results/p2d/grid.json`、`tune.jsonl`。

| 方法 | 調的 hp | 階段 1 點數 | 邊界延伸點數 | 合計 hp 點 | 備註 |
|---|---|---|---|---|---|
| DOB | fc | 8 | 0 | 8 | 最佳 fc=16，不在邊界 |
| LOC-M | lam、prior | 8 | 2 | 10 | lam 停在下邊界（0.998） |
| CEN-ATT（CON-ATT 共用） | lam、prior | 8 | 1 | 9 | prior 停在上邊界（2），但全格 RMSE 2.85–2.86 cm，不敏感 |
| EKF-L | q_b、r_scale | 8 | 2 + **4（使用者核准的偏離 p2d_prespec，r_scale 延伸到不在邊界）** | **14** | **有延伸**；`docs/PROGRESS.md`「EKF-L 延伸調參」 |
| EKF-L+GEO | q_b、r_scale、use_dc | 8 | 2 | **10** | **有延伸**；在 DOB／LOC-M／CEN-ATT 之後才加入，參考 EKF-L 的經驗 |

- 預算不對稱要揭露：EKF-L 14 點、GEO 10 點，多於 DOB 8 點，所以 EKF-L 與 GEO 比 DOB 多了調參機會（對「分享收益」是保守方向：本地基線被調得更強）。所有調參都在 N=3、moderate×1.0 下，因此 DOB 的 fc=16 在 s=0 吃虧。
- 已知調參瑕疵（要在論文揭露）：LOC-M 的 lam 停在下邊界；CEN-ATT 的 prior 停在上邊界；DOB 的 fc=16 與 LOC-M 的 hp 是在 s=1 調的（影響 H2 的解讀，§1.1）。

## 4. 主要量與次要量

評估函式 `run_episode_p2c(method, traj, seed, profile, speed, hp, eta_comp, n)`（`src/smpc/eval/episode_p2c.py`），回傳 dict；每集一列存 jsonl。

- **主要量：RMSE_eff（cm）**
  - 定義：令 `r` = dict 的 `rmse`（payload 位置對參考位置的全程 RMS，單位 m）。
    `RMSE_eff = 100 cm` 若（`finite` 為 False，或 `r` 非有限，或 `success` 為 False，或 `r > 1 m`）；否則 `RMSE_eff = 100·r` cm。
    其中 `success = finite 且 最後 1 s 平均誤差 < 0.10 m 且 最大誤差 < 0.5 m`。
  - 1 m 截斷規則：失敗或發散的集一律記為 1 m = 100 cm（上限，不丟棄、不插補）。這個截斷讓單一崩潰集對均值的影響有上界。
  - 與 pilot 的差異：P2d 的 `rm()` 只對不有限或 >1 m 記 1 m，`success=False` 但 rmse ≤ 1 m 的集保留原值（`experiments/p2d_run.py`）。本文件依使用者指示改為「失敗也記 1 m」。**敏感度分析**：用 P2d 規則重算主表與 F1，並列出受影響的集數。
  - **成功率另外報**（不併入 RMSE_eff 的檢定）：`success` 的比例，Wilson 95% CI。
- **次要量**（只報點估計與 bootstrap CI，不做 Holm，不影響決策）：
  1. 成功率。
  2. 0.5 s 張力預測誤差 `tpe10`（RMS，N；`episode_p2c.py` 的 `lead=50`）。
  3. **ζ̂ 誤差 `zeta_rms_h2`**：後半段（含）所有機的 ζ̂_i − (k_true,i − 1) 的 RMS（程式已輸出；另有全程版 `zeta_rms_all`）。DOB、HYB 無 ζ̂（HYB 有 ζ̂ 估計器，但屬描述性），記 n/a。分享估計相對於 GEO、EKF-L 的 ζ̂ 誤差差值，作為「估計交換改善了什麼」的旁證。
  4. 輔助：`dm_err@5`、`dc_err@5`、`clip_frac`、`slack_events`、`max_tilt_deg`。
- 每條軌跡的單位是 seed；不把兩軌跡平均後檢定（pilot 的 pooled 只用於估 sd）。

## 5. 統計方法

- **配對與 seeds：**同 (s, speed, 軌跡) 內，不同方法用同一 seed；差值以 seed 為單位。
  **確認過程式：**seed 決定 (1) 真值 θ 與 k_true（`draw_truth`：`default_rng([seed,1])`）、初始條件（`[seed,2]`）、感測雜訊（`[seed,3]`、`SensorModel` 的 `[seed,4]`、`[seed,5]`）、CON-ATT 掉包流（`[seed,6]`）；(2) **MPPI 的亂數也由 seed 決定**：`SuspendedMPPITeam` 對機 i 建立 `torch.Generator().manual_seed(seed*100+i)`（`suspended_mppi.py`、`mppi.py`），每步抽樣形狀固定。所以同 seed 的不同方法共用同一組真值、感測流與 MPPI 抽樣流（共同亂數），配對成立；閉迴路軌跡一旦分歧，同一亂數流實際作用到的狀態會不同，所以配對降低的是真值與雜訊層級的變異，不是逐步完全相同。
  不同 s 之間同 seed 也共用亂數流，只有雜訊強度與延遲不同，所以 H2、H4 的「每 seed 斜率」是合理的重複量測。兩軌跡共用 seeds，所以兩軌跡的結果不視為獨立重複（限制）。
- **顯著性檢定：配對差的 sign-flip permutation，單尾。**
  - 實作：對長度 n=40 的配對差向量 x（cm），產生 B = **100,000** 個隨機 ±1 向量（`numpy.random.default_rng(0)` 產生一張 B×n 的矩陣，所有檢定共用），翻號後的均值為 m_b。單尾 p（H0: 均值 ≥ 0，備擇: 均值 < 0）= (1 + #{b: m_b ≤ mean(x)}) / (B + 1)。p 的下限 1/100,001。
  - 需要「均值 > 0」的檢定（H2 斜率）把 x 換成 −x 再用同一公式。**不做雙尾檢定當決策依據**；雙尾 p 只在報告中另列（= min(1, 2·單尾 p) 的上界，僅供參考）。
  - **Holm 一律使用單尾 p。**
- **信賴區間：**對 40 個 seed（配對單位）有放回重抽，重複 10,000 次（`default_rng(0)`），每次取配對差的均值，取 2.5% 與 97.5% 百分位為 95% CI；TOST 的輔助區間取 5% 與 95%（90% CI）。相對效應（差 / 分母方法的均值）用同一組重抽索引計算。bootstrap 的抽樣單位是 seed（同一個索引同時套到兩個方法與同一軌跡）；兩條軌跡各自獨立分析，不在軌跡間抽樣。
- **TOST（H3）：**界限 ±0.25 cm。p_TOST = max(p₁, p₂)，p₁ = 對 x − 0.25 做單尾 sign-flip（H0: 均值 ≥ 0.25），p₂ = 對 −(x + 0.25) 做單尾 sign-flip（H0: 均值 ≤ −0.25）。等價成立需 TOST 的 Holm 後 p < 0.05 **且** 90% CI ⊂ (−0.25, 0.25)；`tests.csv` 的 `rejected` 欄對 TOST 檢定即指「兩條件都滿足」。**±0.5 cm 為敏感度分析**：同樣算 TOST 與 90% CI，並列在報告中，不進 Holm、不影響 H3 的結論；p 為**未校正**，`rejected` = 未校正 p < 0.05 且 90% CI ⊂ (−0.5, 0.5)。
- **H2 / H4 的斜率：**每個 seed、每條軌跡，把 5 個 s 的 y(s) 對 x = s 的數值（0, 0.5, 1, 1.75, 2.5）做 OLS，斜率 = Σ(s_k − s̄)·y_k / Σ(s_k − s̄)²（s̄ = 1.15，分母 3.95）。H2 的 y = B_DOB(s)；H4 的 y = Da(s) − Db(s)。對 40 個斜率做 §5 的 sign-flip。選這個而非 Jonckheere–Terpstra：五個 s 用同一批 seeds，是重複量測，JT 假設獨立組。
- **Holm 家族**（每個家族內 α=0.05；家族之間不再校正；全部用單尾 p）：

| 家族 | 內容 | 檢定數 | 角色 |
|---|---|---|---|
| **F1（主要）** | H1a × 5 s × 2 軌跡，H1b × 5 s × 2 軌跡 | 20 | 主要 |
| **F2（主要）** | H2：DOB 斜率 × 2 軌跡 | 2 | 主要 |
| **F3（主要）** | H3：TOST（s=1，speed 1.0）× 2 軌跡 | 2 | 主要 |
| F4 | H2 的 LOC-M 斜率 × 2 軌跡（B_LOCM(s) = RMSE_eff(LOC-M) − RMSE_eff(CON-ATT)） | 2 | 次要 |
| F5 | H4 斜率 × 2 軌跡 | 2 | 次要 |
| F6 | H3 的 TOST 在 s ∈ {0, 0.5, 1.75, 2.5} × 2 軌跡 | 8 | 次要 |
| F7 | speed 1.5（s=1）的 H1a、H1b × 2 軌跡 | 4 | 次要 |
| F8 | CON-ATT(p_drop=0.5) − EKF-L（s=1，speed 1.0）× 2 軌跡 | 2 | 次要 |

  - **「兩條軌跡都要通過」是交集檢定**：F1 內 20 個檢定各自 Holm；某 (假設, s) 成立 = 該 s 的兩個檢定都被拒絕。交集規則比單一檢定更保守（同時過兩個的機率不大於過一個），不需要再校正。F2、F3、F4、F5、F6、F7 同理。
  - 其餘比較（CON-ATT − EKF-L 的總收益、CON-ATT 對 DOB／LOC-M 在各 s 的差、次要量、G5）：只報點估計與 bootstrap CI，標為描述性，不下顯著結論。
- **檢定力與計畫依據：**見附錄 B 與 `results/prereg/power.md`（由 `experiments/prereg_power.py` 產出，只讀 pilot 資料）。結論：n=40 時 H1a 在 s≥0.5 與 H1b 在 s≤1 有足夠 power；**H1b 在 s≥1.75 的 power 約 0.1**，預期寫成「證據不足」而非「無效」。H2、H3 power ≈ 1。

## 6. 決策規則（每種結果組合怎麼寫）

軌跡標籤：兩條軌跡結論一致才寫成該 s 的結論；不一致就分軌跡寫，不平均。

| 結果 | 結論寫法 |
|---|---|
| H1a 在 s 成立 | 「在 s，分享隊友幾何（相對只用本地資訊的 EKF-L）有可測收益」，同時報相對值（Da / RMSE(EKF-L)）。再標 Δ 層：點估計 ≤ −Δ 寫「達實質門檻」；否則寫「統計上可偵測，但小於事先定的實質門檻」 |
| H1b 在 s 成立 | 「在 s，已有隊友幾何之後，分享估計還有可測的額外收益」；Δ 層同上 |
| H1a 或 H1b 在 s 不成立、CI 包含 0，或 CI < 0 但未過 Holm | 「證據不足」，不寫無效（判定一律依 §1.2 真值表） |
| 只有一條軌跡成立 | 「單軌成立」：分軌跡敘述，不算主張成立 |
| CI 完全 > 0 | 「反向」：前者較差；照實報告，不改 hp 重跑 |
| H1a 與 H1b 的成立 s 不同（例如 H1a 在高 s 成立、H1b 在低 s 成立） | 逐 s 列表，**描述**「收益來源隨 s 變化」；是否寫成趨勢，只看 H4（F5）。不得事後宣稱「分界在 s=…」，除非該 s 前後兩格的符號與 CI 都改變 |
| H4 成立 | 「幾何收益相對於估計交換收益隨 s 增加」（斜率方向），不寫成因果或機制 |
| H4 不成立 | 「來源變化的方向沒有被支持」，只保留逐 s 點估計的描述 |
| H2 成立（F2） | 「簡單本地方法 DOB 的總分享收益平均隨 s 增加」；附 s 軸揭露（§1.1） |
| H2 不成立 | 「收益不隨雜訊增加」；主線第一句降為描述性 |
| H2 的 LOC-M（F4） | 只作次要；成立與否獨立報告，與 DOB 不一致時分開寫 |
| H3 成立（±0.25） | 「2 次迭代的分散式估計交換與集中式在 ±0.25 cm 內等價」；±0.5 cm 敏感度只報告 |
| H3 不成立 | 報告 D3 與 CI，說明分散化有損；H1b 仍以 CON-ATT 為準 |
| 掉包（F8、G4） | 只寫 p=0.5：CON-ATT(p=0.5) 是否仍優於 EKF-L；另列 vs EKF-L+GEO 的描述性差值（GEO 沒有掉包，是有利的對手）；p=0.9 只在描述性段落，附 ζ̂／clip 指標，並註明過度自信 |
| speed 1.5（F7） | 次要；hp 未在 speed 1.5 調，寫成「同 hp 在較快軌跡的表現」 |
| HYB／ORACLE | 只描述 |
| 主線的總結句 | 只由 F1 的逐 s 結果與 H2 組合而成：寫出「在哪些 s，分享隊友幾何有幫助／分享估計有額外幫助」，以及對簡單本地方法的總收益趨勢；不寫 F1 沒檢驗的因果 |

## 7. 偏離政策

- **算偏離**：凍結後對下列任何項目的改變：假設、格子、seeds、方法／hp、RMSE_eff 定義、檢定與次數、Holm 家族、Δ、TOST 界限、排除任何集；以及程式改動而影響 none／est 路徑的結果。
- **不得依 seeds ≥3000 的結果修改分析腳本**（含檢定、CI、家族、圖表、標籤規則）。正式分析腳本 `experiments/prereg_analyze.py` 與本文件在凍結時**一併 commit**，commit hash 寫在文件頂端；凍結後腳本只允許修「不改變輸出數值」的 bug（先證明在固定輸入上輸出逐位相同，並記入 PROGRESS「偏離」）。腳本在 pilot 資料（seeds 2200–2219）上先做 dry run，確認能跑通，這發生在凍結之前。
- **處理**：未改變結果的修 bug（如 est 發散診斷那類）先證明 none 路徑逐位相同（抽樣重跑 RMSE 相同）；之後寫在 `docs/PROGRESS.md` 的「偏離」小節（日期、原因、影響的格子、改前改後）。論文附錄列出全部偏離。偏離後新增的分析一律標為 exploratory，不進 F1–F3。
- 不允許：看過 seeds ≥3000 的結果後再調 hp、換 Δ、換檢定、挑軌跡、刪集。任何集（含崩潰集）都保留，記 1 m。
- 可續跑與重跑：`experiments/prereg_run.py` 以 key 去重續跑；因 crash／OOM（exit 137）中斷的集直接重跑，結果以 key 去重，不算偏離，但要記錄次數。

## 8. 已知限制（要寫進論文）

- 感測 profile 的數值是**假設值**，不是實測；各機雜訊、偏差**彼此獨立**（共模雜訊列為未來的敏感度分析）。
- s 軸不是單一線性尺度（§1.1）；DOB 與 LOC-M 的 hp 在 s=1 調。
- **台數（N）：**本研究只做 N=3。**N≠3 時 attitude 姿態模型會壞掉**：N=4、6 的 smoke 中，CON-ATT 與集中式 CEN-ATT 都出現 clip_frac ≈ 0.98–0.99、估計器失效，EKF-L 則正常（`docs/PROGRESS.md`「PREREG 前程式修正與 GEO 對照」）；N=6 的偽逆在對稱繩配置下數值奇異。N=2 時 Kabsch 只有兩個掛點，旋轉繞兩點連線不可定，也排除。台數 sweep（舊 G3）因此**移出本 PREREG**；擴成 full paper 時需要先修姿態模型再做台數縮放。論文不得宣稱結果推廣到其他台數。
- CON-ATT 為了姿態與力矩偽量測，每步**多轉傳 N−1 = 2 輪位置／量測封包**，這是共識迭代之外的額外通訊（`src/smpc/adapt/con_att.py`）。EKF-L+GEO 同樣轉傳位置與繩向。
- 無風；單一幾何（payload 半徑、繩長、編隊）；只有模擬，無硬體。
- 本地方法**沒有 load cell**，GPAC（Hajieghrary 等，arXiv 2607.00024）不能直接比。
- 調參預算不對稱（§3.3），且全在 N=3、moderate×1.0 下調。
- 兩條軌跡共用 seeds，**不做跨軌跡推論**。
- LOC-M 與 EKF-L 的結論依賴本地基線的選擇；Da、Db 依賴「先幾何、後估計」的順序，不是機制分解。
- 附錄 B 的檢定力用 20 個 pilot seeds 重抽，GEO 在 s=0.5、1.75 靠內插，只是規劃用。

## 9. 運算預算

| 區塊 | 集數 |
|---|---|
| G1：6 方法 × 5 s × 2 軌跡 × 40 seeds | 2400 |
| G2：6 方法 × 2 軌跡 × 40（s=1，speed 1.5） | 480 |
| G4：p=0.5：2 軌跡 × 40；p=0.9：2 × 20 | 120 |
| G5：HYB（2 速度 × 2 軌跡 × 40）＋ ORACLE（2 軌跡 × 40） | 240 |
| **合計** | **3240 集**（舊版 3120：加入 EKF-L+GEO 使 G1 +400、G2 +80，移除台數 sweep −360） |

每集的實測：P2d sweep 在 3 個 worker 同時跑時 `wall_s` 平均 10–12 s（DOB 10.0、LOC-M 11.2、EKF-L 11.7、CEN-ATT 11.8、CON-ATT 12.2、EKF-L+GEO 8.4）；乾淨的單集約 6–7 s（`CLAUDE.md`）。取 11 s：3240 × 11 s ≈ 35,600 s ≈ **9.9 CPU 小時**；樂觀（6.5 s）為 5.9 CPU 小時。`prereg_run.py` 預設最多 3 個 worker，約 **2–3.5 小時牆鐘**；每個 worker 約 190 MB，先看 `free -h`，保持 `available` ≥ 8 GB，外面加 `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0`。結果寫 `results/prereg/episodes.jsonl`（可續跑）。

## 10. 評估腳本規格（給沒看過對話的人）

### 10.1 輸入

`results/prereg/episodes.jsonl`：每行一集，是 `run_episode_p2c` 回傳的 dict 加上 `key`、`tag`（= "prereg"）、`wall_s`。欄位：`method`、`profile`（`"s=<值>"`，例如 `"s=1.75"`）、`traj`（`circle`／`fig8`）、`speed`、`seed`、`eta_comp`、`hp`（dict）、`rmse`（m）、`success`、`finite`、`tpe10`、`zeta_rms_h2`、`dm_err@5`、`dc_err@5`、`clip_frac`、`slack_events`、`max_tilt_deg` 等。

- **執行器：`experiments/prereg_run.py`**（自帶執行器，邏輯同 `p2c_common.run_jobs`：最多 3 worker、依 `available` 記憶體降載、以 key 去重續跑；`--dry` 只列工作數與前 5 個）。**不用 `p2c_common.run_jobs`**，因為它的 `_work` 有 `assert seed < 3000` 的保護（防止調參期誤用正式 seeds）且不傳 `n`；不改共用程式以免動到 none／est 路徑。job = `dict(key, tag, method, profile, speed, traj, seed, hp, eta_comp, n=3)`。
- **key 格式**：`prereg|<method>|<profile>|<traj>|<speed>|<seed>|<hp 字串>`（speed 寫成 `1.0`／`1.5`；hp 字串為 `k=v` 依字母序以逗號相接，`%g` 格式）。**分析不靠 key 配對**，只用 key 去重（同一 key 重複時取第一筆）。
- **索引**：`(method, profile, speed, p_drop, traj, seed)`，其中 p_drop 取自 `hp.get("p_drop", 0)`。
- **hp 比對規則**：只取 hp 與 §3.2 相同的列：去掉 `p_drop` 後，鍵集合與數值須與凍結 hp 完全相同；CON-ATT 的 hp 缺 `iters` 視為 2、缺 `p_drop` 視為 0（pilot 資料沒記這兩項）。正式 CON-ATT 的 hp 為 `{lam, prior, iters=2, p_drop}`。HYB 依 speed 取 §3.2 的 fc、tau。
- **jsonl 欄名 → csv 欄名**：`dm_err@5` → `dm_err5_mean`、`dc_err@5` → `dc_err5_mean`、`tpe10` → `tpe10_mean`、`zeta_rms_h2` → `zeta_rms_h2_mean`、`clip_frac` → `clip_frac_mean`（皆為格內均值，略過 n/a）；`rmse` + `success` + `finite` → `rmse_eff_*`、`success_rate`。

### 10.2 步驟

1. 對每集算 `RMSE_eff`（§4）。
2. **配對差**：對每個檢定，在固定 (s, speed, p_drop, traj) 下，取兩個方法在各 seed 的 RMSE_eff，相減 → 長度 40 的向量 x（cm）。缺 seed 或某格不足 40 → 腳本報錯停止（不得靜默刪集）。腳本另有 `--allow-partial`（略過缺格並列出），**只供 dry-run，正式分析不得使用**。
3. **斜率**：H2、H4 見 §5，每 seed 先算 5 個 s 的 y，再 OLS。
4. **檢定**：§5 的單尾 sign-flip（B=100,000，`default_rng(0)`）、Holm 在各家族內（用單尾 p）、bootstrap 95% CI（10,000 次，`default_rng(0)`）、TOST（90% CI）。
5. **標籤**：依 §1.2 與 §6 輸出每個 (假設, s) 的結論標籤（成立／不成立；Δ 層；證據不足／反向）。
6. **敏感度分析**：(a) P2d 的 `rm()` 規則重算 F1，列出受影響集數；(b) H3 用 ±0.5 cm；(c) 排除 `success=False` 的集重算 F1：**成對刪除**（任一方法在該 seed 失敗，該 seed 的配對整對刪除），報告每個檢定的 n，只報不當結論。主分析仍把失敗集記 1 m。

**實作細節（寫死）：**
- sign-flip 矩陣：`numpy.random.default_rng(0).integers(0, 2, (B, n)) * 2 - 1`（每個 n 一張，同 n 的檢定共用）；翻號均值以矩陣乘法計算。p 公式中「m_b ≤ mean(x)」比較時，容差為 `1e-9 · max(1, max|x|)`（把浮點誤差視為相等，算入 ≤）。
- bootstrap 索引：`default_rng(0).integers(0, n, (10000, n))`，同 n 的所有 CI（95%、90%、相對效應）共用；百分位用 `numpy.percentile` 預設內插。
- `cells.csv` 的 `rmse_eff_ci_*`：該格各 seed 的 RMSE_eff 做上述 bootstrap（單格，非配對）的 95% 百分位 CI；`cells.csv` 涵蓋所有 hp 相符的格（含 G2、G4、G5）。
- `rel_effect` = 配對差均值 / **B 欄方法的 RMSE_eff 均值**（Da 的分母是 EKF-L，Db 的分母是 EKF-L+GEO）；斜率檢定與 DESC 的 slope 列為空。
- 描述性列（`family = DESC`，不進 Holm，`p_*` 與 `rejected` 為空）：B_DOB(s)、B_LOCM(s)、總收益 CON-ATT − EKF-L（各 s、兩軌跡）、CON-ATT(p=0.5／0.9) 對 EKF-L+GEO 與 EKF-L 的差。
- `cells_hyp.csv` 的 `established` = 兩軌跡 `rejected` 皆真；`evidence_label` 依 §1.2 真值表（空 = 成立；「單軌成立(軌跡)」「反向(軌跡)」「證據不足」「未等價」）。`delta_layer`／`delta_label` 只對 H1a/H1b 填。
- ζ̂ 誤差相對 GEO、EKF-L 的差值：腳本只在 `cells.csv` 列各格均值，不另出差值表（旁證用，人工對照）。

### 10.3 輸出（`results/prereg/`）

| 檔名 | 內容 |
|---|---|
| `cells.csv` | 欄位：`method, profile, speed, p_drop, traj, n, rmse_eff_mean_cm, rmse_eff_ci_lo, rmse_eff_ci_hi, success_rate, succ_ci_lo, succ_ci_hi, tpe10_mean, zeta_rms_h2_mean, dm_err5_mean, dc_err5_mean, clip_frac_mean`。每格一列 |
| `tests.csv` | 欄位：`family, test_id, hyp, s, speed, traj, A, B, n, mean_diff_cm, ci95_lo, ci95_hi, ci90_lo, ci90_hi, rel_effect, p_onesided, p_holm_adj, rejected, delta_layer` |
| `cells_hyp.csv` | 欄位：`hyp, s, circle_rejected, fig8_rejected, established, delta_label, evidence_label`（每個 (假設, s) 一列） |
| `h2_h4_slopes.csv` | 每 seed、軌跡、假設的斜率與 5 個 s 的 y |
| `sensitivity.csv` | §10.2 步驟 6 的結果 |
| `PREREG_results.md` | 依 §6 組好的結論文字與表（由腳本產生，不手改） |
| `fig_H1_by_s.png`、`fig_H2_B_DOB.png` | Da(s)、Db(s)（兩條軌跡，含 CI）與 B_DOB(s) 的圖 |

### 10.4 玩具範例

n=4 個 seed 的配對差（cm）x = [−0.6, −0.3, −0.7, −0.2]，均值 −0.45。窮舉 16 種翻號，只有全負的翻號和 ≤ −1.8，單尾 p = 1/16 = 0.0625（隨機版本 B 大時趨近此值）。若這是 F1 內 20 個 p 中最小的一個：Holm 第 1 步門檻 0.05/20 = 0.0025，0.0625 > 0.0025，所以不拒絕，Holm 逐步停止，其餘檢定都不拒絕。
Holm 小例（3 個檢定的家族）：p = [0.001, 0.02, 0.04] 排序後門檻 0.0167、0.025、0.05，三個都拒絕；p = [0.001, 0.03, 0.04] 則 0.001 拒絕，0.03 > 0.025 停止，其餘不拒絕。
RMSE_eff：某集 `success=False`、`rmse = 0.30 m` → 記 100 cm（P2d 規則會保留 30 cm）；`success=True`、`rmse = 0.0286 m` → 2.86 cm。
斜率：某 seed 的 B_DOB 在 s = (0, 0.5, 1, 1.75, 2.5) 為 (0.3, 0.5, 1.0, 2.0, 3.0) cm，s̄ = 1.15，分子 = Σ(s−s̄)B = 4.43，分母 3.95，斜率 = 1.12 cm / 單位 s。
TOST：x 的均值 −0.03 cm、sd 0.08，n=40：p₁（H0: 均值 ≥ 0.25）與 p₂（H0: 均值 ≤ −0.25）都極小，p_TOST = 較大者；同時要 90% CI ⊂ (−0.25, 0.25) 才算等價。

## 11. 已決議（使用者，2026-10-30）

1. 主線：分享值不值得、該分享什麼，取決於本地方法的強度與感測品質。
2. 主要假設為 H1a（知道隊友位置）、H1b（再交換估計），5 個 s × 2 軌跡，單尾 sign-flip；成立 = 兩軌跡都過。
3. H2 只用斜率檢定，「點估計非遞減」改描述性，並揭露 s 軸非線性與 hp 在 s=1 調。
4. H3：TOST ±0.25 cm；±0.5 cm 為敏感度分析。
5. Δ = 0.5 cm 為實質門檻；檢定 0 與 Δ 分開寫。
6. 台數 sweep 移出 PREREG（N≠3 姿態模型壞掉；N=2 排除），列入限制。
7. 每格 40 seeds（3000–3039）；p=0.9 為 20 seeds，描述性。
8. HYB、ORACLE 描述性。
9. 結果路徑 `results/prereg/`；凍結時由使用者 commit，hash 寫入本文件。
10. RMSE_eff 對失敗集記 1 m；ζ̂ 誤差為次要量（`zeta_rms_h2`，程式已輸出）。

**使用者已於 2026-10-31 凍結時確認以下三點（照預設）：**
- (a) Holm 家族的切分（F1 = 20 個 H1a/H1b 檢定；F2、F3 各自獨立；其餘為次要）。
- (b) H4 列為次要假設（F5）。
- (c) G2（speed 1.5）擴成 6 個方法並列為次要家族 F7。

---

## 附錄 A：先前 pilot 摘要（**已看過的資料，不作為證據**）

出處：`results/p2d/P2d_report.md`、`results/p2d/geo_report.md`、`docs/PROGRESS.md`、`results/p2c/P2c_locm.md`。seeds 2020–2029（調參／底線）、2100–2119（P2c）、2200–2219（P2d 掃描與 GEO 拆解）。

- **P2c（moderate/harsh，exploratory）**：CEN-ATT(est) − LOC-M(est) 四格 −1.2～−2.5 cm（CI 全 <0）；LOC-M 補償本身省 2.4–6.8 cm；HYB(est) 在 moderate 與 CEN-ATT(est) 持平、harsh 輸 0.7–1.2 cm。分享對 ζ̂ 的差模誤差減半，但共模 ζ̄ 與 δm 仍共線（ρ≈0.92）（`results/p2c_ident/`）。
- **P2d 調參分數（cm，moderate×1.0）**：DOB 4.09、LOC-M 3.93、EKF-L 3.33、EKF-L+GEO 3.25、CEN-ATT 2.85。
- **P2d 感測掃描（兩軌跡平均 RMSE，cm）**：s=0：DOB 2.83、LOC-M 3.76、EKF-L 3.03、CEN-ATT 2.50、CON-ATT 2.48；s=2.5：8.64／6.65／4.68／3.82／3.78。CEN-ATT − EKF-L：−0.53、−0.68、−0.64、−0.67、−0.86 cm（沒有分界）；對 DOB 的差 −0.33 → −4.83 cm，對 LOC-M −1.26 → −2.84 cm（隨 s 擴大）。CON−CEN −0.02～−0.05 cm。
- **GEO 拆解（pilot，s ∈ {0, 1, 2.5}，seeds 2200–2219，n=20/格，兩軌跡 pooled，cm [95% CI]）**（`results/p2d/geo_report.md`）：RMSE（EKF-L／GEO／CON-ATT）：s=0 3.03／3.04／2.48；s=1 3.54／3.03／2.86；s=2.5 4.68／3.89／3.78。
  - 隊友幾何的增量 Da（GEO − EKF-L）：s=0 +0.01 [−0.28, +0.31]；s=1 −0.51 [−0.82, −0.19]；s=2.5 −0.78 [−1.23, −0.36]。
  - 估計交換的增量 Db（CON − GEO）：s=0 −0.55 [−0.84, −0.29]；s=1 −0.17 [−0.30, −0.04]；s=2.5 −0.11 [−0.48, +0.35]。
  - 總收益 CON − EKF-L：−0.55／−0.68／−0.90。
  - 這是 H1a、H1b 與 H4 的來源；**pilot 觀察值**：s=1 時總收益約 19%（0.68 / 3.54）。這個比例是 pilot 觀察值，不是預期值或門檻。
- **掉包（CON-ATT，s=1）**：p=0／0.5／0.9 的 RMSE 2.86／2.83／2.65 cm；p=0.9 時估計誤差更差，疑似過度自信，不引用。
- 以上沒有一項被當作正式結論；正式結論只來自 seeds 3000–3039 並依本文件分析。

## 附錄 B：檢定力（規劃用，非證據）

來源：`results/prereg/power.md`（`experiments/prereg_power.py`，只讀 `results/p2d/sweep.jsonl`；seeds 2200–2219 的 20 個 seed 有放回重抽到 n=40，2000 次模擬，sign-flip 5000 次，Holm 在家族內，兩軌都過）。EKF-L+GEO 只有 s ∈ {0, 1, 2.5}，s=0.5、1.75 用 GEO RMSE 對 s 線性內插（†，粗估）。

**as-observed 的 power（兩軌都過，Holm 後）：**

| 假設 | s=0 | s=0.5† | s=1 | s=1.75† | s=2.5 |
|---|---|---|---|---|---|
| H1a | 0.013（觀察到的效應 ≈ 0，這格預期不成立） | 0.87 | 0.93 | 0.98 | 0.99 |
| H1b | 1.00 | 0.85 | **0.69** | **0.09** | **0.11** |

- H2（DOB 斜率）1.00；H2（LOC-M，F4）1.00；H3（TOST ±0.25，s=1）1.00；H4（F5）1.00。sd×1.5 時：H1a 在 s=1 為 0.50、H1b 在 s=1 為 0.25、H4 為 0.95，H2、H3 仍 ≥ 0.997。
- 「兩軌點估計 ≤ −0.5 cm」的機率（Δ 層）：H1a 在 s=1 為 0.47、在 s=2.5 為 0.96；H1b 在 s=0 為 0.49，s≥0.5 幾乎為 0。也就是說，**H1b 即使成立，預期多半小於 Δ**。
- 真均值平移（保守 α=0.05/20，兩軌都過）：H1b 在 s=1，真均值 −0.2 cm 為 0.84、−0.1 cm 為 0.13；H1b 在 s=2.5，−0.3 cm 為 0.28、−0.5 cm 為 0.59。H1a 在 s=2.5，−0.3 cm 為 0.10、−0.5 cm 為 0.50。
- 限制：20 個 seed 重抽到 40，有效資訊仍是 20；觀察到的均值本身有抽樣誤差，as-observed 可能偏樂觀（sd×1.5 列為偏保守的對照）；GEO 的 s=0.5、1.75 是內插。
- 結論：H1b 在 s≥1.75 預期偵測不到（power ≈ 0.1），應事先接受它會被寫成「證據不足」。
