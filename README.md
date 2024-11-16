# Gemma Continued Pre-training

このリポジトリは、Google's Gemmaモデルの継続事前学習を実装したものです。H100×8 GPUを使用した大規模な分散学習に最適化されています。

## 環境構築

必要な依存パッケージのインストール：

```bash
pip install -r requirements.txt
```

また、Weights & Biasesのアカウントが必要です：

```bash
wandb login YOUR_WANDB_KEY
```

## プロジェクト構造

```
gemma-pretraining/
├── src/
│   ├── train_deepspeed.py  # メインの学習スクリプト
│   └── utils.py            # 補助関数
├── configs/
│   ├── train_configs/
│   │   └── train_base.yaml # 学習パラメータの設定
│   └── deepspeed/
│       └── ds_config_zero2.json  # DeepSpeed設定
├── scripts/
│   └── run_pretraining.sh  # 実行用シェルスクリプト
├── outputs/                 # モデルの出力ディレクトリ
└── requirements.txt
```

## 主な設定パラメータ

`configs/train_configs/train_base.yaml`で以下の主要なパラメータを設定できます：

- `model`: モデルとトークナイザーの設定
  - `max_length`: 最大シーケンス長（デフォルト：2048）
- `train`: 学習関連の設定
  - `learning_rate`: 学習率
  - `num_train_epochs`: エポック数
  - `per_device_train_batch_size`: GPUあたりのバッチサイズ
  - その他のDeepSpeedやWandB関連の設定

## 実行方法

SLURMクラスタ上での実行：

```bash
sbatch scripts/run_pretraining.sh
```

## 学習の監視

学習の進捗はWandBダッシュボードで確認できます：
https://wandb.ai/[YOUR_WANDB_USERNAME]/gemma-continued-pretraining

## 出力

モデルのチェックポイントは`outputs`ディレクトリに保存されます：
- 学習済みモデル
- トークナイザー
- 学習ログ

## ハードウェア要件

- GPUs: 8x NVIDIA H100
- メモリ: GPU当たり最低80GB
- ストレージ: 最低500GB（データセットとチェックポイント用）

## 引用

このプロジェクトを引用する場合は、以下のBibTeXエントリを使用してください：

```bibtex
@misc{gemma2024,
  author = {Google},
  title = {Gemma: Open Models Based on Gemini Research},
  year = {2024},
  publisher = {Google},
  url = {https://blog.google/technology/developers/gemma-open-models/}
}
```

## ライセンス

このプロジェクトはApache License 2.0の下でライセンスされています。

## 注意事項

- H100 GPUに最適化された設定になっています
- 学習時間は3GBのデータセットで約数時間を想定しています
- より大きなデータセットを使用する場合は、バッチサイズやその他のパラメータの調整が必要になる場合があります

## トラブルシューティング

よくある問題と解決方法：

1. OOMエラー
   - バッチサイズを小さくする
   - gradient_accumulation_stepsを増やす
   - max_lengthを調整する

2. DeepSpeedの初期化エラー
   - 環境変数`MASTER_ADDR`と`MASTER_PORT`が正しく設定されているか確認
   - 各ノードでDeepSpeedが正しくインストールされているか確認

## コントリビューション

1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## 開発者

- @あなたのGitHubアカウント

## 謝辞

- Google Gemmaチーム
- DeepSpeedチーム
- Hugging Faceチーム

上記のREADMEは必要に応じて環境やプロジェクトの詳細を追加・修正してください。特に以下の項目は要修正です：

- WandBのユーザー名
- GitHubアカウント
- 特定の環境変数やパス
- ライセンス情報（必要に応じて）
