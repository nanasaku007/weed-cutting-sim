# 次のセッションへの引き継ぎ

更新：2026-10-19。ROADMAPの実装順番1〜3が完了し、次は**順番4：BC再学習と条件別の閉ループ評価**です。このセッションでは保存・pushまでとし、順番4の学習はまだ実行していません。

## 最初に読むもの

1. [ROADMAP.md](ROADMAP.md)：目標と実装単位。末尾の「実装単位と優先順位」の順番4を進める。
2. [STAGE3_REVIEW.md](STAGE3_REVIEW.md)：教師改善に加え、観測の版分けと退避・回復検査を追加した理由。
3. [STAGE3_RESULTS.md](STAGE3_RESULTS.md)：確定した教師比較、データ、失敗と制約。
4. [README.md](README.md)：セットアップ、CLI、再現コマンド。

## 完了した内容

- 順番1・2：3背景、3カメラ、3植物形状×0/10/20度。追加背景は非接触の装飾。茎全体の保護判定は `stem_segment_v1`。背景比較・幾何検証は [STAGE1_2_RESULTS.md](STAGE1_2_RESULTS.md) に記録。
- 順番3：`expert.py` に経路と鋏の開き幅を選ぶルールベース教師 `waypoint_v2` を追加。旧教師 `direct_v1` と旧15次元入力・checkpointは維持。
- 新観測 `state30_v2` は旧15次元＋対象の茎方向3＋保護対象2本の根元/上端の工具相対位置12。仕様の正本は `environment.py` の `OBS_SCHEMAS`。
- `stage3.py` で教師比較・回復開始・デモ収集・事前固定splitを実装。`run.py` は観測版の保存と次元検査、データ指定splitでの学習に対応。
- 新教師の通常評価264/270、回復評価270/270。各30配置を9条件で再使用しており、270個の独立した配置ではない。**この数字はBCの性能ではない。**
- 450試行から445成功軌道・53,824フレームを収集。学習356軌道/43,269フレーム、検証89軌道/10,555フレーム。回復89軌道（学習71/検証18）。
- 自動テスト43件を整備。旧BCの通常評価29/30を維持し、旧評価とのepisode結果一致も確認済み。

## ローカル成果物とGitHubの境界

コード・文書・集計JSONはGit管理。`outputs/`、動画、データ、重み、`vendor/`、仮想環境は `.gitignore` で除外する。同じローカル作業フォルダでセッションを切り替えれば既存成果物を利用できるが、新規cloneには含まれない。

| パス | 内容 |
| --- | --- |
| `outputs/stage3/demos_state30_v1.npz` | 次のBC学習に使うデータ |
| `outputs/stage3/demos_state30_v1.json` | 全450試行、条件、split、hashのmanifest |
| `outputs/stage3/demos_state30_v1.attempts.json` | 収集試行記録 |
| `outputs/stage3/dataset_validation.json` | 実データ検証結果 |
| `outputs/stage3/evaluation_iid.json` | 通常条件の両教師比較 |
| `outputs/stage3/evaluation_recovery.json` | 回復条件の両教師比較 |
| `outputs/stage3/legacy_bc_regression.json` | 旧BCの回帰確認 |
| `outputs/stage3/teacher_waypoint.mp4` | 開発seed 42006の教師成功例。学習BCの動画ではない |
| `outputs/bc_v2.npz` | 旧15次元BC。上書きしない |
| `experiments/stage3_summary.json` | Git管理する集計・hash・失敗seed |

データSHA-256：

```text
64f6835f89d611cfb9bb18fa538e963f3e271b79c9deeac76c4c97a3afc32d12
```

Python 3.12 / MuJoCo 3.3.7で検証。既存の作業フォルダではリポジトリから見て `../../.venv-cutting/bin/python` が利用できる。新規cloneではREADMEに従って `.venv` と外部Pandaアセットを用意する。以下は適切な仮想環境を有効化した状態で、リポジトリ直下から実行する。

## 次の作業

1. Gitの状態とデータの存在を確認し、データを検証する。既存データがある場合は450試行を取り直さない。

   ```bash
   python validate_demos.py outputs/stage3/demos_state30_v1.npz
   ```

   データがない場合のみ、READMEの `stage3.py collect --seed 60000 --episodes 50` で再生成し、manifestとhashを確認する。

2. 新しい出力先に30次元BCを学習する。学習は未実施なので、結果を推測して報告しない。

   ```bash
   python run.py train --data outputs/stage3/demos_state30_v1.npz --out outputs/stage4/bc_state30_v1.npz
   ```

   データに保存した356/89軌道のsplitを使用し、フレーム単位で再分割しない。教師のphaseラベルは解析用で、入力へ混ぜない。

3. 3形状×3傾斜、方位0/90/180/270度を循環する条件別BC評価を整備する。**現在の `stage3.py evaluate` は教師専用**。その結果をBC評価として扱わない。`run.py evaluate --policy bc` と新checkpoint、`--observation-schema state30_v2 --geometry stem_segment_v1` を使うか、同じrolloutを使う集計runnerを追加する。回復評価には `stage3.py` の初期化処理を再利用する。

4. 同条件の教師と比較し、成功、誤切断、保護対象侵入、地面接触、時間切れ、所要時間を記録する。BCの行動に教師の経路修正を暗黙に挟まない。旧15次元モデルとの比較は共通タスクの範囲を明示する。

5. 条件別結果と失敗を文書・集計JSONへ残す。順番5の90〜120秒デモは、BCの実測結果が揃ってから進める。

## seedと評価の扱い

| seed範囲 | 用途 |
| --- | --- |
| 20000・30000番台 | 初回BCの通常/OOD。既に分析済み |
| 41000・42000・42100番台 | 開発・プレビュー |
| 51000〜51029 | コード固定後の順番3通常評価 |
| 52000〜52029 | コード固定後の順番3回復評価 |
| 60000〜60449 | 今回の学習/検証デモ収集 |

51000・52000番台の結果は既に確認済み。今後のモデル選択・調整に使うなら開発/回帰用とし、最終評価には新seedを別途確保する。例えば70000・71000番台が候補だが、まだ実行・予約していないため、使用前に既存成果物との重複を確認する。評価の最大180 step（9秒）や判定を変更する場合は比較条件の変更として明記する。

## 残る課題・誤解しやすい点

- 新教師も完全ではない。通常評価の51017・51029は20度/方位90度で時間切れ（3形状に共通）。収集の60135は退避で保護対象侵入。60222・60287・60434は時間切れ、60442は候補内で安全経路を発見できず終了。5失敗は学習配列から除外し、manifestには保持。
- 成功デモだけのBCが未知状態から回復できる保証はない。検証の予測誤差だけで完了とせず、閉ループで測る。
- X位置・高さを広げる旧OOD分布、任意の株間隔、葉全体やアーム全体の衝突回避は今回の正式比較に含めていない。`no_safe_route` は候補探索の失敗で、経路が存在しない証明ではない。
- 状態入力BCであり、画像認識・VLA・事前学習モデルのfine-tuning・実機制御は未実装。対象座標と切断状態はシミュレータから与える。
- GUIはMacの `mjpython`、動画描画にはグラフィックスセッションが必要。学習・数値評価はGUI不要。

## 次セッションへ渡す依頼文

> HANDOFF.mdとROADMAP.md、STAGE3_REVIEW.md、STAGE3_RESULTS.mdを読み、ROADMAPの実装順番4（BC再学習・条件別評価）を進めて。既存の30次元データと固定splitを確認して使い、旧checkpointは保存して。教師とBCの結果を区別し、別seedの閉ループ評価と失敗分析まで記録して。
