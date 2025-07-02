# 機械学習・画像認識学習用コード集

このリポジトリは機械学習の画像認識を学習するための実践的なコード集です。初心者から中級者まで段階的に学習できるように構成されています。

## 📁 ディレクトリ構造

```
machine-learning/image-recognition/
│
├── basic/                    # 基礎的な画像認識
│   ├── cifar10_cnn.py       # CIFAR-10 CNNモデル
│   └── mnist_classifier.py   # MNIST手書き数字認識
│
├── advanced/                 # 高度な手法
│   └── transfer_learning.py  # 転移学習による画像分類
│
├── utils/                    # ユーティリティ
│   └── custom_dataset.py     # カスタムデータセット処理
│
├── datasets/                 # データセット保存場所
├── models/                   # 学習済みモデル保存場所
│
├── requirements.txt          # 必要なライブラリ
└── README.md                # このファイル
```

## 🚀 セットアップ

### 1. 必要なライブラリのインストール

```bash
cd /home/smitsu32/projects/machine-learning/image-recognition
pip install -r requirements.txt
```

### 2. GPU使用の場合（オプション）

CUDA対応のGPUを使用する場合：

```bash
pip install tensorflow-gpu>=2.12.0
```

## 📚 学習コード一覧

### 基礎編

#### 1. MNIST手書き数字認識 (`basic/mnist_classifier.py`)

機械学習の「Hello World」とも言われるMNIST手書き数字認識です。

**特徴:**
- 28x28ピクセルのグレースケール画像
- 0-9の数字分類（10クラス）
- 60,000枚の学習データ、10,000枚のテストデータ
- シンプルなMLPとCNNの両方を実装

**実行方法:**
```bash
python basic/mnist_classifier.py
```

**学習内容:**
- 基本的なニューラルネットワーク構造
- 畳み込みニューラルネットワーク（CNN）
- データの前処理と正規化
- モデルの学習と評価
- 結果の可視化

#### 2. CIFAR-10画像分類 (`basic/cifar10_cnn.py`)

カラー画像の分類タスクです。

**特徴:**
- 32x32ピクセルのカラー画像
- 10クラス分類（飛行機、車、鳥、猫、鹿、犬、カエル、馬、船、トラック）
- 50,000枚の学習データ、10,000枚のテストデータ

**実行方法:**
```bash
python basic/cifar10_cnn.py
```

**学習内容:**
- カラー画像の処理
- より複雑なCNNアーキテクチャ
- 過学習防止（Dropout、Early Stopping）
- 学習率スケジューリング

### 応用編

#### 3. 転移学習 (`advanced/transfer_learning.py`)

事前学習済みモデルを使用した効率的な画像分類です。

**特徴:**
- VGG16、ResNet50、MobileNetV2などの事前学習済みモデル
- 少ないデータでも高精度を達成
- ファインチューニング機能

**実行方法:**
```bash
python advanced/transfer_learning.py
```

**学習内容:**
- 転移学習の概念と実装
- 事前学習済みモデルの活用
- ファインチューニング戦略
- データ拡張（Data Augmentation）

#### 4. カスタムデータセット (`utils/custom_dataset.py`)

独自のデータセットを使用した画像認識です。

**特徴:**
- フォルダ構造からの自動データセット作成
- データ拡張機能
- 学習/検証/テスト分割
- バッチ処理

**実行方法:**
```bash
python utils/custom_dataset.py
```

**学習内容:**
- 実際のプロジェクトでのデータ準備
- データ拡張の実装
- カスタムデータローダー
- データの可視化と分析

## 🎯 学習の進め方

### 初心者向け

1. **MNIST手書き数字認識**から始める
   - 機械学習の基本概念を理解
   - TensorFlow/Kerasの基本操作を習得

2. **CIFAR-10画像分類**に進む
   - カラー画像の処理を学習
   - より実践的なCNNアーキテクチャを体験

### 中級者向け

3. **転移学習**を試す
   - 効率的な学習手法を習得
   - 実用的なアプリケーション開発の基礎

4. **カスタムデータセット**で実践
   - 独自のプロジェクトに応用
   - データ前処理の重要性を理解

## 📊 実行例とベンチマーク

### MNIST（CNN）
- 学習時間: 約5-10分（CPU）
- 期待精度: 99%以上
- メモリ使用量: 約1GB

### CIFAR-10（CNN）
- 学習時間: 約20-30分（CPU）
- 期待精度: 70-80%
- メモリ使用量: 約2GB

### 転移学習（MobileNetV2）
- 学習時間: 約10-15分（CPU）
- 期待精度: 85-90%
- メモリ使用量: 約3GB

## 🔧 カスタマイズ・拡張

### パラメータ調整

各スクリプトには以下のようなパラメータがあります：

```python
# 学習パラメータの例
epochs = 20           # エポック数
batch_size = 32       # バッチサイズ
learning_rate = 0.001 # 学習率
```

### モデルアーキテクチャ

CNNの層構成を変更して実験できます：

```python
# モデル構成の例
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # ... 他の層
])
```

## 🐛 トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - バッチサイズを小さくする
   - 画像サイズを小さくする
   - 不要なデータをメモリから削除

2. **学習が収束しない**
   - 学習率を調整する
   - バッチサイズを変更する
   - データの前処理を見直す

3. **GPUが認識されない**
   - CUDA、cuDNNが正しくインストールされているか確認
   - tensorflow-gpuが正しくインストールされているか確認

### デバッグ方法

```python
# TensorFlowのGPU使用状況確認
print("GPU利用可能:", tf.config.list_physical_devices('GPU'))

# メモリ使用量確認
import psutil
print(f"メモリ使用量: {psutil.virtual_memory().percent}%")
```

## 📈 発展的な学習

### 次のステップ

1. **物体検出**: YOLO、R-CNNなどの手法
2. **セマンティックセグメンテーション**: U-Net、DeepLabなど
3. **生成モデル**: GAN、VAEによる画像生成
4. **強化学習**: ゲームAIなどへの応用

### 推奨リソース

- **書籍**: 「ゼロから作るDeep Learning」シリーズ
- **オンライン**: Coursera、edX、Udacityの機械学習コース
- **論文**: arXiv.orgでの最新研究論文
- **データセット**: Kaggle、UCI Machine Learning Repository

## 🤝 貢献・フィードバック

このコード集は学習目的で作成されています。改善提案や質問がある場合は、以下の方法でお気軽にご連絡ください：

- Issues・Pull Requestを作成
- コードレビューのリクエスト
- 新しい学習サンプルの提案

## 📄 ライセンス

このコードは教育目的で自由に使用できます。商用利用の際は、使用するライブラリのライセンスを確認してください。

---

**Happy Learning! 🎉**

機械学習の画像認識の世界へようこそ！このコード集があなたの学習の役に立つことを願っています。