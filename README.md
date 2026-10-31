# 選択的な雑草切断

ロボットアームに取り付けた鋏で、3本の植物から指定された1本を切断するMuJoCoシミュレーションです。隣の植物を残し、切断後に開始位置まで戻る一連の動作を、模倣学習で扱います。

Pandaアームと開閉1自由度の鋏を使用し、ルールベース教師のデモからExtraTrees回帰によるBehavior Cloning（BC）を学習します。学習・評価はCPUで実行できます。

## 評価結果

30次元の状態入力を使うBCと、デモ生成に使った教師を比較しました。

| 開始条件 | 学習BC | ルールベース教師 |
| --- | --- | --- |
| 通常 | 243/270（90.0%） | 267/270（98.9%） |
| 回復（閉じた鋏・位置ずれあり） | 229/270（84.8%） | 267/270（98.9%） |

細葉・広葉・背高の3形状と、0°・10°・20°の3傾斜を組み合わせた評価です。各開始条件で同じ30配置を9条件に使っているため、270件は延べ試行数です。

成功条件は、指定対象だけを切断し、保護対象への工具侵入や工具の地面接触がなく、開始位置から12 mm未満まで帰還することです。制限時間は9秒。BCには接近・退避中の侵入と帰還の時間切れが残っています。

条件別の結果と失敗例は [BCの評価結果](STAGE4_RESULTS.md)、評価設定は [stage4_protocol.json](experiments/stage4_protocol.json)、集計は [stage4_summary.json](experiments/stage4_summary.json) に記載しています。

## セットアップ

Apple Silicon Mac、Python 3.12.14、MuJoCo 3.3.7で動作確認しています。描画にはOpenGL環境が必要です。

```bash
git clone https://github.com/nanasaku007/weed-cutting-sim.git
cd weed-cutting-sim
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python fetch_assets.py
```

`fetch_assets.py` はMuJoCo MenagerieのPandaモデル（約31 MB）を固定commitから取得し、ライセンスと各ファイルのSHA-256を保存します。モデルの出典は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) を参照してください。

データ・学習済みモデル・動画は、以下のコマンドで `outputs/` に生成します。`requirements-lock.txt` に検証時の依存パッケージのバージョンを固定しています。

## 動作確認

30次元観測と茎全体の保護判定を使う教師を起動します。MacのGUIには `mjpython` を使います。

```bash
mjpython play.py --policy expert --teacher waypoint_v2 --observation-schema state30_v2 --geometry stem_segment_v1 --plant broad --tilt 10
```

| キー | 操作 |
| --- | --- |
| R | 次の配置 |
| スペース | 停止・再開 |
| ウィンドウを閉じる | 終了 |

手動操作は `mjpython play.py --policy manual` で起動します。W/SでX方向、A/DでY方向、E/Qで上下に移動し、C/Oで鋏を閉じる・開く操作ができます。

## データ収集・学習・評価

仮想環境を有効にした状態で、リポジトリ直下から実行します。

```bash
# 3形状×3傾斜、各50配置。20%は回復開始
python stage3.py collect --seed 60000 --episodes 50 --out outputs/stage3/demos_state30_v1.npz
python validate_demos.py outputs/stage3/demos_state30_v1.npz --out outputs/stage3/dataset_validation_stage4.json

# データに保存した軌道単位の分割でBCを学習
python run.py train --data outputs/stage3/demos_state30_v1.npz --out outputs/stage4/bc_state30_v1.npz --seed 42

# 同じ配置で教師とBCを比較
python stage4.py --checkpoint outputs/stage4/bc_state30_v1.npz --seed 70000 --episodes 30 --out outputs/stage4/evaluation_iid.json
python stage4.py --checkpoint outputs/stage4/bc_state30_v1.npz --seed 71000 --episodes 30 --recovery --out outputs/stage4/evaluation_recovery.json

# 学習済みBCの再生
mjpython play.py --policy bc --checkpoint outputs/stage4/bc_state30_v1.npz --observation-schema state30_v2 --geometry stem_segment_v1 --plant broad --tilt 10

# テスト
python -m pytest -q
```

評価は描画なしで実行できます。`stage4.py` は既存の評価JSONへの上書きを拒否します。学習コマンドは同名のモデルを上書きするため、別の実験では出力先を変更してください。

収集450試行のうち、成功445軌道・53,824フレームを学習用データに採用しました。学習356軌道、検証89軌道に分け、回復デモはそれぞれ71軌道・18軌道です。ExtraTreesは48本、最大深さ22、葉の最小サンプル数2とし、最終評価前に設定を固定しました。

収集・検証の詳細は [教師とデータ収集の結果](STAGE3_RESULTS.md) に、保存済み評価の検査・集計処理は `stage4_report.py` にあります。評価seedをモデルの調整に使う場合、その後の評価には別のseedを使います。

## 観測と制御

```text
シミュレータの状態（30次元）
  → BC：工具のXYZ移動量・鋏の開き率
  → 指令制限・IK・関節位置制御
  → MuJoCo
  → 次の状態
```

| 観測 | 次元 |
| --- | --- |
| 対象切断点と工具の相対位置 | 3 |
| 工具の世界位置 | 3 |
| 鋏の開き・対象の切断状態 | 2 |
| アームの関節角 | 7 |
| 対象の茎方向 | 3 |
| 保護対象2本の根元・先端と工具の相対位置 | 12 |

制御周期は20 Hz、物理ステップは2 msです。行動はXYZ各軸の移動量（最大8 mm/制御ステップ）と鋏の目標開き率（0〜1）の4次元。工具の水平姿勢はIKで維持します。BCの実行時に教師の経路補正は加えません。

観測仕様は `environment.py` の `OBS_SCHEMAS` に定義しています。現在のモデルは `state30_v2`、初回のモデルは `state15_v1` です。モデルと観測仕様が一致しない場合はエラーになります。

データは観測・行動・次観測・episode ID・seedをNumPy形式で保存します。学習済みモデルも木構造を数値として保存し、推論時にはNumPyだけを使います。

## 背景と植物のプレビュー

背景は `baseline`、縁石際の `curb`、植樹帯の `tree`、河川緑地の `river` を選択できます。植物は `slender`（細葉）、`broad`（広葉）、`tall`（背高）で、傾斜は0〜20度、方位も指定できます。

```bash
# 初回モデル bc_v2.npz を使った背景・カメラのプレビュー
python preview.py --out outputs/roadmap_preview

# 植物形状と傾斜のプレビュー
python preview.py --mode plants --out outputs/plants_preview

# 植樹帯・広葉型・20度を教師で再生
mjpython play.py --policy expert --teacher waypoint_v2 --observation-schema state30_v2 --geometry stem_segment_v1 --scene tree --plant broad --tilt 20 --azimuth 0
```

細葉・広葉の上部の茎は14 cm、背高型は20 cmです。3本には同じ傾斜と方位を適用します。カメラは全景・接写・真上の3種類を用意しています。

## シミュレーションの範囲

切断は、茎と有限長の刃が交差し、鋏の角度が閉じ方向に閾値を跨いだときの幾何イベントとして扱います。切断位置は事前に設定した接合部から茎に沿って12 mm以内に限定し、上部を固定するweld制約を解除して落下させます。

保護判定は鋏の包絡と保護茎の線分の交差を検出します。侵入や工具の地面接触を検出すると、その試行を停止します。

- 対象の指定・座標・切断状態はシミュレータから取得します。画像認識や知覚誤差は評価対象外です。
- 切断力、茎や葉の変形、刃の切れ味、任意位置での破断はモデル化していません。
- 葉やアーム全体との衝突、実機での動作・安全性は未検証です。
- 追加背景は描画用です。樹木や護岸との接触、水の物理は扱いません。評価背景は `baseline` に固定しています。

## 実験記録

| 記録 | 内容 |
| --- | --- |
| [初回評価](RESULTS.md) | 15次元BCの通常条件・位置と高さを広げた条件での比較 |
| [背景・植物の拡張](STAGE1_2_RESULTS.md) | 背景3種、植物3形状、傾斜・切断・着地の検証 |
| [教師の設計](STAGE3_REVIEW.md) | 経路・開き幅の選択と30次元観測の設計 |
| [教師評価・データ収集](STAGE3_RESULTS.md) | 教師の比較、回復デモ、軌道単位のデータ分割 |
| [BCの評価](STAGE4_RESULTS.md) | 通常・回復条件の成績、失敗分析、検証誤差 |

### 初回モデルの再現

15次元観測と切断点付近の保護判定を使った初回評価のコマンドです。上記の30次元モデルとは出力先を分けています。

```bash
python run.py collect --episodes 240 --out outputs/demos_v2.npz
python run.py train --data outputs/demos_v2.npz --out outputs/bc_v2.npz
python run.py evaluate --policy bc --checkpoint outputs/bc_v2.npz --episodes 30 --seed 20000 --out outputs/bc_iid.json --logs outputs/logs/bc_iid
python run.py evaluate --policy bc --checkpoint outputs/bc_v2.npz --episodes 30 --seed 30000 --ood --out outputs/bc_ood.json --logs outputs/logs/bc_ood
python run.py evaluate --policy expert --episodes 30 --seed 20000 --out outputs/expert_iid.json
python run.py evaluate --policy expert --episodes 30 --seed 30000 --ood --out outputs/expert_ood.json
python run.py video --policy bc --checkpoint outputs/bc_v2.npz --episodes 5 --seed 20000 --out outputs/learned_cutting.mp4
```

Physical AI 応用1講座の最終課題として制作しました。FK／IK、模倣学習、閉ループ評価、実験の再現性管理を、選択的な除草タスクに組み合わせています。
