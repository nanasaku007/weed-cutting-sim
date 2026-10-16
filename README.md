# 選択的雑草切断 — Mac / MuJoCo MVP

Pandaアームに固定した鋏で、3本の茎から指定された1本を切り、開始位置へ退避します。**実機とColabは不要**で、デモ生成からCPU学習、閉ループ評価、動画生成まで実装しています。

今回のAIは **状態入力のBehavior Cloning（BC）** です。ExtraTrees回帰モデルが教師の動きを学習し、毎ステップの手先移動と鋏の開閉を予測します。既存モデルのfine-tuning、画像認識、VLAはまだ実装していません。

## 現在地と今後の方針

- [評価結果](RESULTS.md)：通常条件29/30成功、位置・高さを広げた条件16/30成功。
- [開発方針](ROADMAP.md)：道路・河川を模した背景、植物形状、傾き、撮影構成の拡張計画。
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

## 次の実験

まず3種類の背景と3種類の植物外観でデモの表現を改善し、次に茎の傾き・配置・密度を物理判定にも反映します。その後に教師経路と学習データを改善し、画像入力へ進みます。具体的な実装順序、完了条件、90〜120秒の動画構成は [ROADMAP.md](ROADMAP.md) に記載しています。

## 講義との接続・出典

基礎編第2回のFK／IKとMuJoCo、第3回のBCと閉ループ評価、第6回のepisode分割、第7回のログと実行制約、応用編第2回の再現性管理を小さなタスクへ適用しています。元モデルと公式ドキュメントの参照先は `THIRD_PARTY_NOTICES.md` に記載しています。

YouTubeへの動画アップロードはまだ行っていません。講義資料や個人用のファイルはこのリポジトリに含みません。
