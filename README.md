# 選択的雑草切断 — Mac / MuJoCo MVP

Pandaアームに固定した鋏で、3本の茎から指定された1本を切り、開始位置へ退避します。**実機とColabは不要**で、デモ生成からCPU学習、閉ループ評価、動画生成まで実装しています。

今回のAIは **状態入力のBehavior Cloning（BC）** です。ExtraTrees回帰モデルが教師の動きを学習し、毎ステップの手先移動と鋏の開閉を予測します。既存モデルのfine-tuning、画像認識、VLAはまだ実装していません。

## 現在地と今後の方針

- [次セッションへの引き継ぎ](HANDOFF.md)：順番3までの到達点、ローカル成果物、次のBC再学習・評価の手順。
- [評価結果](RESULTS.md)：通常条件29/30成功、位置・高さを広げた条件16/30成功。
- [開発方針](ROADMAP.md)：道路・河川を模した背景、植物形状、傾き、撮影構成の拡張計画。
- [背景・植物拡張の結果](STAGE1_2_RESULTS.md)（2026-10-18）：3背景・3視点、3形状×3傾斜を実装。既存BCを固定した背景比較と、新しい茎全体の保護判定での教師評価を記録。
- [順番3の方針レビュー](STAGE3_REVIEW.md) / [教師改善・データ収集の結果](STAGE3_RESULTS.md)：保護対象を考慮する教師、30次元の新観測、回復を含むデモ収集。
- このリポジトリにはコード、依存関係、テスト、ドキュメントを含めています。動画、データ、学習済み重み、外部モデルは含めず、下記コマンドで取得・再生成します。

## ローカルで生成される成果物

- `outputs/learned_cutting.mp4`：学習モデルが5配置を操作する動画。各配置は全景と接写で同じ試行を再生し、双方ともシミュレーション時間の1倍速。
- `outputs/bc_iid.json` / `outputs/bc_ood.json`：学習モデルの通常条件／範囲外条件の評価。
- `outputs/expert_iid.json` / `outputs/expert_ood.json`：同じ配置でのルールベース教師との比較。
- `RESULTS.md`：測定結果と限界。

金色の根元マーカーが対象、青色が保護対象です。モデルには対象の正確な座標が入力されます。色を画像から認識しているという意味ではありません。

## セットアップと最初の動作

Python 3.12とOpenGL環境を用意します。Apple Silicon Mac、Python 3.12.14、MuJoCo 3.3.7で確認済みです。

```bash
git clone https://github.com/nanasaku007/weed-cutting-sim.git
cd weed-cutting-sim
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python fetch_assets.py

# 学習前に教師の動きを確認
mjpython play.py --policy expert
```

privateリポジトリのcloneには閲覧権限が必要です。MacのGUI viewerには `python` ではなく `mjpython` を使います。`R` で次の配置、スペースで停止／再開、ウィンドウを閉じると終了します。学習・評価はGUIを起動しません。

手動で動かす場合：

```bash
mjpython play.py --policy manual
```

| キー | 操作（世界座標） |
| --- | --- |
| W / S | X方向へ移動 |
| A / D | Y方向へ移動 |
| E / Q | 上 / 下へ移動 |
| C / O | 鋏を閉じる / 開く |
| R | 次の配置へリセット |
| スペース | 一時停止 / 再開 |

手動操作はキー入力ごとに短い移動指令を与えるデバッグ用です。

取得する外部データは公式Pandaモデル約31 MBだけです。取得先commitを固定し、ライセンスとファイルごとのSHA-256も保存します。初回取得後、学習・評価はオフラインで実行できます。`requirements.txt` は直接の依存関係、`requirements-lock.txt` は検証時の間接依存も含む固定版です。

以下のコマンドは仮想環境を有効化した状態で実行します。既存の開発環境を使う場合も、その環境を有効化すれば同じコマンドを利用できます。

## デモ生成 → 学習 → 評価

```bash
# 240配置から成功デモを収集（失敗もJSONに残す）
python run.py collect --episodes 240 --out outputs/demos_v2.npz

# episode単位で80/20に分割し、CPUでBCを学習
python run.py train --data outputs/demos_v2.npz --out outputs/bc_v2.npz

# 学習と別seedの通常条件、30試行
python run.py evaluate --policy bc --checkpoint outputs/bc_v2.npz --episodes 30 --seed 20000 --out outputs/bc_iid.json --logs outputs/logs/bc_iid

# 通常範囲からX +35 mm、切断高さ +25 mmに移した条件、30試行
python run.py evaluate --policy bc --checkpoint outputs/bc_v2.npz --episodes 30 --seed 30000 --ood --out outputs/bc_ood.json --logs outputs/logs/bc_ood

# 教師を同じ配置で評価
python run.py evaluate --policy expert --episodes 30 --seed 20000 --out outputs/expert_iid.json
python run.py evaluate --policy expert --episodes 30 --seed 30000 --ood --out outputs/expert_ood.json

# 学習モデルの実動作動画を出力（Macの描画機能を使用）
python run.py video --policy bc --checkpoint outputs/bc_v2.npz --episodes 5 --seed 20000 --out outputs/learned_cutting.mp4

# 学習した方策をGUIで再生
mjpython play.py --policy bc --checkpoint outputs/bc_v2.npz

# 切断判定、実際のrollout、reset、入力検査など
python -m pytest -q
```

動画と評価レポートは異なる出力名にしてください。各コマンドは指定した同名の成果物を更新します。

## 実装の境界

以下は初回ベースライン（CLIの既定値）の説明です。新しい背景・植物・判定の指定方法は「背景と植物のプレビュー」を参照してください。

| 部分 | 実装 |
| --- | --- |
| アーム | 公式Pandaモデル。関節位置actuatorを使って `mj_step` で動かす。episode開始時以外に関節位置を直接移さない |
| 鋏 | 固定刃＋回転する刃、開閉1自由度。ばねなし |
| 茎 | 円柱状の単純形状＋楕円体の葉。地面から切断点までの根元と、上部を分離 |
| 切断 | 茎が有限長の刃の切断領域にあり、鋏角が閉じ方向へ0.12 radを跨ぐと切断イベント |
| 切断後 | 上部を固定するweld制約を解除。重力で落下し地面に接触する |
| 教師 | 位置誤差に応じた接近、位置合わせ後の閉鎖、鋏を閉じた退避、十分離れた後の再開放 |
| AI | 成功デモの状態→4次元actionをExtraTrees 48本で教師あり学習 |
| 下位制御 | 姿勢を水平に保つヤコビアンIK、関節目標の変化制限、位置追従、シミュレーション上の重力補償 |
| 評価 | 指定対象のみ切断し、保護対象侵入・工具の地面接触なしで開始位置へ戻ると成功 |

切断力、茎のしなり、刃の切れ味、任意高さでのメッシュ分割は再現していません。切断点はepisode開始時に設定され、上下の茎の接合位置になります。AIが任意の位置で本物の草を切れるという実証ではありません。

## 観測・行動・データ

- 20 Hzの制御、0.002秒の物理stepを25回進めて1 action。
- 観測15次元：対象切断点と工具の相対位置3、工具の世界位置3、鋏の開き1、対象切断状態1、アーム関節角7。
- 相対位置のみ10倍（dm）にスケール、位置はm、関節角はrad。キー・順番は `environment.py` の `OBS_NAMES` が正本。
- 行動4次元：XYZ移動（各軸最大8 mm/制御step）と鋏の目標開き率（0〜1）。工具の姿勢はIK側が維持。
- 正確な対象位置・対象ID・切断状態はシミュレータの状態から与える。この段階では知覚誤差を評価しない。
- `demos_v2.npz`：observation、action、next_observation、episode_id、seed。画像は含まない。現段階ではLeRobotDatasetではなく、軽量なNumPy形式。
- 学習／検証はepisode単位で分割。評価seedは20000番台と30000番台で独立。
- `bc_v2.npz`：推論に必要な木構造と数値だけ。pickleを読み込まず、NumPyだけで推論可能。
- 評価ログ：観測、モデルが出したaction、実際のactuator指令、IK残差を各stepで保存。

成功率以外に、誤切断、保護対象の切断点付近への工具侵入、工具の地面接触、所要時間、切断高さ誤差を記録します。保護判定は茎の切断点付近の幾何判定で、葉全体や人との衝突回避を保証するものではありません。

## 背景と植物のプレビュー

ROADMAPの順番1・2を小さく実装しました。`baseline` に加え、`curb`（縁石際）、`tree`（植樹帯）、`river`（河川緑地）を選べます。追加背景は**装飾のみで接触しません**。旧来の床・土台・縁石の接触は維持しています。背景の樹木や護岸を避ける動作を実証したものではありません。

```bash
# 既存のbc_v2.npzを固定し、同じseedを3場面で撮影する。
# 9植物条件は現行教師で動かす。どちらも失敗を含めJSONへ記録する。
python preview.py --out outputs/roadmap_preview

# 各条件30試行の開発評価（描画しない）。プレビューの1例と区別する。
python preview.py --episodes 30 --no-render --out outputs/stage1_2_evaluation

# 植物の確認だけなら学習済みcheckpointは不要
python preview.py --mode plants --out outputs/plants_preview

# 広葉型・20度の傾斜をGUIで確認
mjpython play.py --policy expert --scene tree --plant broad --tilt 20 --azimuth 0 --geometry stem_segment_v1

# 同じ条件で動画／評価も可能（出力名は旧評価と分ける）
python run.py evaluate --policy expert --scene tree --plant broad --tilt 20 --geometry stem_segment_v1 --seed 41000 --episodes 30 --out outputs/broad20_development.json
```

`preview.py` の既定seedは41000。描画にはmacOSのグラフィックスセッションが必要です。GUI接続がない環境でも `--no-render` で評価できます。プレビュー動画は各場面に冒頭・末尾の静止画を1秒ずつ置き、その間はシミュレーション時間の1倍速です。最終提出用の90〜120秒動画は後続段階です。

| 設定 | 選択肢・意味 |
| --- | --- |
| `--scene` | `baseline` / `curb` / `tree` / `river` |
| `--plant` | `legacy` / `slender`（細葉型）/ `broad`（広葉型）/ `tall`（背高型） |
| `--tilt` / `--azimuth` | 傾斜0〜20度、傾く方位。現在は3本に同じ角度を適用 |
| `--geometry legacy_point_v1` | 旧来の点による判定。再現用の既定値。傾斜は不可 |
| `--geometry stem_segment_v1` | 茎線分と切断面の交差、茎全体の保護判定、傾斜時の固定制約 |
| `render(camera=...)` | `overview` / `closeup` / `top`。同一試行の前後で固定するカメラ |

新条件でも観測15次元・行動4次元は同じです。切断点の世界座標を揃え、地表の根元を移動して傾斜を作ります。新しい判定でも分離位置はあらかじめ決めた接合部で、切断面の交点が接合部から茎に沿って12 mm以内の場合だけ切れます。任意位置でのメッシュ分割ではありません。

細葉・広葉の上部の茎は14 cm、背高型は20 cm。葉は工具と接触しませんが、主要な葉の形状変更は剛体の慣性にも反映されます。したがって9植物条件は背景だけの比較とは分けています。新しいデータを集める際は旧データと異なる出力名を使い、JSONの `environment.geometry` と条件を保存してください。順番1・2の時点ではデータの追加収集・BC再学習は行わず、後述の順番3でデータを追加収集しました。BC再学習は次の順番4です。

## 次の実験

順番3の教師改善とデモ収集を実装しました。次は順番4のBC再学習・閉ループ評価です。新しい30次元入力は旧checkpointと混在できません。教師の成功率がそのままBCの成功率になるわけではないため、モデル学習後に別途評価します。

## 改善した教師と新データ（順番3）

`--teacher waypoint_v2` は、正確な茎の座標を使って経路と鋏の開き幅を選ぶルールベース教師です。旧教師は既定の `direct_v1` として残しています。新教師は `--geometry stem_segment_v1` が必要です。

```bash
# 新教師の動きを確認（まだ学習BCではない）
mjpython play.py --policy expert --teacher waypoint_v2 --observation-schema state30_v2 --geometry stem_segment_v1 --scene tree --plant broad --tilt 10 --azimuth 180 --seed 42006

# 3形状×3傾斜×各30配置。同じ配置で旧教師と新教師を比較
python stage3.py evaluate --seed 51000 --episodes 30 --out outputs/stage3/evaluation_iid.json

# 閉じた鋏＋位置ずれから開始する回復条件の比較
python stage3.py evaluate --seed 52000 --episodes 30 --recovery --out outputs/stage3/evaluation_recovery.json

# 各50配置、計450試行。20%は回復開始。成功軌道だけを保存
python stage3.py collect --seed 60000 --episodes 50 --out outputs/stage3/demos_state30_v1.npz
python validate_demos.py outputs/stage3/demos_state30_v1.npz --out outputs/stage3/dataset_validation.json

# 新教師を1条件で評価する場合
python run.py evaluate --policy expert --teacher waypoint_v2 --observation-schema state30_v2 --geometry stem_segment_v1 --plant broad --tilt 20 --episodes 30 --seed 51000 --out outputs/stage3/broad20.json
```

`state30_v2` は旧15次元の後ろに、対象の単位茎方向3値、保護対象2本の「根元・先端－工具」12値を追加します。相対位置の単位は従来どおりdmで、保護対象は元のplant ID順（対象を除外）です。観測の正本は `environment.py` の `OBS_SCHEMAS`。新教師は内部のphaseを行動決定に必要とせず、データの `teacher_phase` は解析用で学習入力には含めません。

新データには `observation_schema`、`condition_id`、`recovery`、事前に決めた `validation_episode_ids` が入ります。`run.py train` はそのsplitを使用し、新checkpointにも観測版を保存します。旧checkpointは15次元として読み込み、違う次元の入力を与えるとエラーにします。旧データや旧モデルは上書きしません。

次の順番4で使う学習コマンドは以下です。この段階では実際のBC再学習はまだ行っていません。

```bash
python run.py train --data outputs/stage3/demos_state30_v1.npz --out outputs/stage4/bc_state30_v1.npz
```

今回評価した51000・52000番台を今後モデルの選択・調整に使う場合、それらは開発用として扱い、改善を結論づける最終評価には新しいseedを確保してください。具体的な実装順序と動画構成は [ROADMAP.md](ROADMAP.md) を参照してください。

## 講義との接続・出典

基礎編第2回のFK／IKとMuJoCo、第3回のBCと閉ループ評価、第6回のepisode分割、第7回のログと実行制約、応用編第2回の再現性管理を小さなタスクへ適用しています。元モデルと公式ドキュメントの参照先は `THIRD_PARTY_NOTICES.md` に記載しています。

YouTubeへの動画アップロードはまだ行っていません。講義資料や個人用のファイルはこのリポジトリに含みません。
