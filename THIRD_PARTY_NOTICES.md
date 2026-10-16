# Third-party model

Pandaモデルは [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35/franka_emika_panda) の `panda_nohand.xml` と参照meshを利用しています。

- Commit: `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`
- モデルのライセンス: Apache-2.0（取得先の `LICENSE` を同梱取得）
- 取得: `fetch_assets.py`。取得した各ファイルのSHA-256は `vendor/franka_emika_panda/manifest.json` に記録。
- 変更点: 元ファイルは保持し、実行時にPythonで鋏、茎、地面、接合制約、カメラ設定を追加。元のkeyframeを外し、タスク側で初期状態を設定。
- 配布時: モデルを同梱する場合は `vendor/franka_emika_panda/LICENSE` とこの通知も含めてください。取得スクリプトだけを公開する場合、初回実行にネットワーク接続が必要です。

MuJoCoのPython APIとMac viewerの実行方法は [公式Pythonドキュメント](https://mujoco.readthedocs.io/en/3.3.7/python.html) を参照しています。
