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
├── docs/                     # 詳細ガイド・理論解説
│   └── AI画像認識総合ガイド.md  # 機械学習の理論と実践の総合解説
│
├── datasets/                 # データセット保存場所
├── models/                   # 学習済みモデル保存場所
│
├── requirements.txt          # 必要なライブラリ
└── README.md                # このファイル
```

## 📖 理論学習ガイド

### 🧠 [AI画像認識総合ガイド](docs/AI画像認識総合ガイド.md)

プログラミングを始める前に、**理論的な背景をしっかりと理解したい方におすすめ**の詳細ガイドです。

**内容:**
- 🔬 **機械学習の基礎理論**: AI、機械学習、ディープラーニングの関係
- 🎯 **コンピュータビジョンの種類**: 画像分類、物体検出、セグメンテーション
- 🏗️ **ニューラルネットワークの仕組み**: パーセプトロンからCNNまで
- 📊 **各手法の数学的背景**: 損失関数、最適化、正則化の詳細
- 🚀 **実用技術の解説**: 転移学習、データ拡張、ハイパーパラメータチューニング
- 💼 **産業応用事例**: 医療、製造業、農業での活用例

**対象者:**
- 機械学習の理論をしっかり理解してから実践したい方
- 数学的背景に興味がある方
- AI技術の全体像を把握したい方

> 💡 **学習のコツ**: 理論ガイドを読んでから実際のコードを動かすと、各手法の「なぜ」と「どのように」が深く理解できます。

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

**手法の概要:**
MNIST（Modified National Institute of Standards and Technology）は、手書き数字（0-9）の画像分類を行う機械学習の入門的なタスクです。この手法では、多層パーセプトロン（MLP）と畳み込みニューラルネットワーク（CNN）の2つのアプローチを比較できます。

- **MLP（多層パーセプトロン）**: 全結合層のみで構成されたシンプルなニューラルネットワーク
- **CNN（畳み込みニューラルネットワーク）**: 画像の局所的な特徴を抽出する畳み込み層を持つネットワーク

機械学習の「Hello World」とも言われ、基本概念を学ぶのに最適です。

**特徴:**
- 28x28ピクセルのグレースケール画像
- 0-9の数字分類（10クラス）
- 60,000枚の学習データ、10,000枚のテストデータ
- シンプルなMLPとCNNの両方を実装

**実行方法:**
```bash
python basic/mnist_classifier.py
```

**学習目標と段階:**

🎯 **第1段階: 基本概念の理解**
- ニューラルネットワークの基本構造
- 活性化関数（ReLU、Softmax）の役割
- 損失関数（categorical_crossentropy）の意味
- 最適化アルゴリズム（Adam）の動作

🎯 **第2段階: データ処理の理解**
- 画像データの正規化（0-255 → 0-1）
- ワンホットエンコーディングの必要性
- 学習/検証データの分割の重要性

🎯 **第3段階: モデル比較**
- 多層パーセプトロン（MLP）: シンプルな全結合層
- 畳み込みニューラルネットワーク（CNN）: 画像特化型
- 各アーキテクチャの精度と学習時間の違い

**期待される学習成果:**
- 精度: 97-99%（CNN使用時）
- 学習時間: 10-15分（CPU）
- 理解できる概念: 基本的なディープラーニングの流れ

**実習のポイント:**
1. 両方のモデル（MLP・CNN）を試して精度を比較
2. エポック数やバッチサイズを変更して結果を観察
3. 混同行列で誤分類のパターンを分析
4. 学習曲線で過学習の有無を確認

#### 2. CIFAR-10画像分類 (`basic/cifar10_cnn.py`)

**手法の概要:**
CIFAR-10（Canadian Institute for Advanced Research）は、10種類のオブジェクト（飛行機、車、鳥、猫、鹿、犬、カエル、馬、船、トラック）を含むカラー画像の分類タスクです。MNISTよりも現実的で複雑な画像認識問題となります。

この手法では、**深い畳み込みニューラルネットワーク（Deep CNN）**を使用します：
- **複数の畳み込み層**: 段階的に複雑な特徴を抽出（エッジ→テクスチャ→形状→オブジェクト）
- **プーリング層**: 画像サイズを縮小し、計算効率を向上
- **正則化手法**: DropoutやEarly Stoppingで過学習を防止

カラー画像（RGB 3チャンネル）の処理を学び、実用的なスキルを身につけられます。

**特徴:**
- 32x32ピクセルのカラー画像（RGB 3チャンネル）
- 10クラス分類（飛行機、車、鳥、猫、鹿、犬、カエル、馬、船、トラック）
- 50,000枚の学習データ、10,000枚のテストデータ
- 現実的な画像認識の課題

**実行方法:**
```bash
python basic/cifar10_cnn.py
```

**学習目標と段階:**

🎯 **第1段階: カラー画像処理の理解**
- RGBチャンネル（3次元）の扱い方
- カラー画像特有の前処理手法
- チャンネル数がモデルに与える影響

🎯 **第2段階: 高度なCNNアーキテクチャ**
- 複数の畳み込み層の積み重ね
- フィルタ数の段階的増加（32→64→64）
- 特徴マップサイズの変化（32×32→16×16→8×8）

🎯 **第3段階: 過学習対策の実践**
- Dropoutによる正則化（50%）
- Early Stoppingの活用
- 学習率スケジューリング（ReduceLROnPlateau）

🎯 **第4段階: モデルの最適化**
- バッチサイズの調整
- エポック数の決定
- GPU使用時のメモリ管理

**期待される学習成果:**
- 精度: 70-80%（基本CNN）
- 学習時間: 20-30分（CPU）、5-10分（GPU）
- 理解できる概念: 実用的なCNN設計パターン

**実習のポイント:**
1. MNISTとの精度・学習時間の違いを体感
2. 各畳み込み層の出力サイズを計算して理解
3. 学習曲線から過学習のタイミングを特定
4. クラス別の分類性能を分析（一部のクラスが難しい理由）

**発展的な実験:**
- データ拡張（回転、反転、ズーム）の効果
- 異なるCNNアーキテクチャの試行
- より深いネットワーク（4-5層）の構築

### 応用編

#### 3. 転移学習 (`advanced/transfer_learning.py`)

**手法の概要:**
転移学習（Transfer Learning）は、大規模データセット（ImageNet：100万枚以上の画像）で事前に学習済みのモデルを、新しいタスクに応用する手法です。ゼロから学習するよりも効率的で高精度な結果を得られるため、実際の業務で最も頻繁に使用されます。

この手法の仕組み：
- **事前学習済みモデル**: VGG16、ResNet50、MobileNetV2などの高性能ネットワーク
- **特徴抽出器として活用**: 下位層は汎用的な画像特徴（エッジ、テクスチャ）を学習済み
- **カスタム分類器の追加**: 上位層に新しいタスク用の分類器を接続
- **2段階学習**: ①固定学習（上位層のみ）→②ファインチューニング（全体の微調整）

少ないデータと短い学習時間で高性能なモデルを構築できる、実用性の高い手法です。

**特徴:**
- VGG16、ResNet50、MobileNetV2などの事前学習済みモデル
- ImageNet（100万枚以上の画像）で事前学習済み
- 少ないデータでも高精度を達成
- 2段階学習（固定学習→ファインチューニング）

**実行方法:**
```bash
python advanced/transfer_learning.py
```

**学習目標と段階:**

🎯 **第1段階: 転移学習の概念理解**
- 事前学習済みモデルの利点（時間短縮、高精度）
- ImageNetデータセットの特徴と汎用性
- 特徴抽出器としてのCNNの活用

🎯 **第2段階: アーキテクチャの比較**
- **VGG16**: 深い畳み込み層、高精度だが重い
- **ResNet50**: 残差接続で勾配消失問題を解決
- **MobileNetV2**: 軽量化、モバイル・エッジデバイス向け

🎯 **第3段階: 2段階学習の実践**
- **第1段階**: ベースモデル固定でトップ層のみ学習
- **第2段階**: ファインチューニングで全体を微調整
- 学習率の段階的調整（0.0001 → 0.00001）

🎯 **第4段階: 高度なデータ拡張**
- RandomFlip（水平反転）
- RandomRotation（回転）
- RandomZoom（ズーム）
- RandomContrast（コントラスト調整）

**期待される学習成果:**
- 精度: 85-95%（CIFAR-10での転移学習）
- 学習時間: 10-15分（CPU）、3-5分（GPU）
- 理解できる概念: 実用的なディープラーニング開発手法

**実習のポイント:**
1. 3つの事前学習済みモデルの性能比較
2. 固定学習とファインチューニングの効果の差
3. データ拡張の有無による精度への影響
4. 学習済みモデルの特徴マップの可視化

**実用的な応用:**
- 医療画像診断（X線、MRI）
- 製品品質検査（不良品検出）
- 農業（作物の病気診断）
- セキュリティ（顔認証、監視カメラ）

**発展的な実験:**
- 異なるデータセット（Fashion-MNIST、STL-10）での転移学習
- カスタムデータでの実践
- アンサンブル学習（複数モデルの組み合わせ）

#### 4. カスタムデータセット (`utils/custom_dataset.py`)

**手法の概要:**
カスタムデータセット処理は、実際のビジネス課題を解決するために独自の画像データを機械学習に活用する手法です。MNISTやCIFAR-10のような標準データセットとは異なり、現実世界の多様で不均質なデータを扱う実践的なスキルです。

この手法で学ぶ技術：
- **データ収集と整理**: フォルダ構造でのクラス分類、画像品質の管理
- **前処理パイプライン**: サイズ統一、色空間変換、正規化の自動化
- **データ分割**: 学習・検証・テスト用の適切な分割とクラスバランス調整
- **データ拡張**: 回転、反転、ズーム等でデータ量を人工的に増加
- **品質管理**: 破損ファイル、ノイズ画像の自動検出と除外

業務レベルのプロジェクトでは必須となる、データハンドリングの総合的なスキルを身につけられます。

**特徴:**
- フォルダ構造からの自動データセット作成
- 複数画像形式への対応（JPG、PNG、BMP等）
- 自動的な学習/検証/テスト分割
- データ拡張機能とバッチ処理

**実行方法:**
```bash
python utils/custom_dataset.py
```

**学習目標と段階:**

🎯 **第1段階: データセット構造の理解**
- 標準的なディレクトリ構造の作成
- ```
  dataset/
  ├── class1/
  │   ├── image1.jpg
  │   └── image2.png
  └── class2/
      ├── image3.jpg
      └── image4.png
  ```
- クラスラベルの自動抽出
- ファイル形式の検証とフィルタリング

🎯 **第2段階: 画像前処理パイプライン**
- PIL（Python Imaging Library）による画像読み込み
- 色空間の統一（RGB変換）
- サイズの統一（リサイズ処理）
- 正規化（0-255 → 0-1）

🎯 **第3段階: データ分割戦略**
- 学習用：60-70%
- 検証用：15-20%
- テスト用：15-20%
- 層化サンプリング（クラス比率を保持）

🎯 **第4段階: データ拡張の実装**
- 幾何学的変換（回転、反転、拡大縮小）
- 色調変換（明度、コントラスト調整）
- ノイズ付加
- バッチ生成とシャッフル

**期待される学習成果:**
- 任意の画像データセットを機械学習用に変換可能
- データ品質の評価と改善手法の習得
- 効率的なデータローダーの実装

**実習のポイント:**
1. 実際の画像データセットを準備して試行
2. クラス不均衡への対処法を実験
3. データ拡張の効果を定量的に評価
4. メモリ効率的なバッチ処理の実装

**実用的な応用例:**
- **製造業**: 製品の良品/不良品判定
- **医療**: 病理画像の自動診断支援
- **農業**: 作物の生育状況や病害の検出
- **小売**: 商品カテゴリの自動分類
- **セキュリティ**: 監視カメラの異常検知

**データ収集のベストプラクティス:**
- 各クラス最低100枚以上の画像
- 多様な撮影条件（照明、角度、背景）
- 高品質な画像（ブレ・ノイズの少ない）
- バランスの取れたクラス分布

**発展的な技術:**
- アクティブラーニング（効率的なデータ収集）
- 弱教師あり学習（ラベルが不完全なデータの活用）
- データの品質評価指標
- 自動ラベリング手法

## 🎯 学習の進め方

### 📖 **レベル1: 機械学習入門（推奨学習期間: 1-2週間）**

#### ステップ1: MNIST手書き数字認識
**目標**: 機械学習の基本概念を身につける

**学習手順:**
1. **環境準備** (30分)
   ```bash
   pip install -r requirements.txt
   python basic/mnist_classifier.py
   ```

2. **基本概念の理解** (2-3時間)
   - ニューラルネットワークの構造を理解
   - 活性化関数、損失関数、最適化の役割を学習
   - モデル選択画面で「1」（MLP）を選択して実行

3. **CNNとの比較** (2-3時間)
   - モデル選択画面で「2」（CNN）を選択して実行
   - MLPとCNNの精度・学習時間の違いを比較
   - 混同行列で誤分類パターンを分析

4. **パラメータ実験** (2-4時間)
   - エポック数を変更（5, 10, 20）して学習曲線を観察
   - バッチサイズを変更（32, 64, 128）して影響を確認
   - 過学習の兆候を学習曲線から読み取る

**習得スキル:**
- ✅ 基本的なニューラルネットワークの理解
- ✅ TensorFlow/Kerasの基本操作
- ✅ 学習曲線の読み方
- ✅ モデル評価の基本

---

### 📊 **レベル2: 実用的CNN（推奨学習期間: 2-3週間）**

#### ステップ2: CIFAR-10画像分類
**目標**: より現実的な画像認識問題を解決する

**学習手順:**
1. **実行と結果確認** (1時間)
   ```bash
   python basic/cifar10_cnn.py
   ```
   - MNISTとの違い（精度、学習時間、難易度）を体感

2. **CNNアーキテクチャの理解** (3-4時間)
   - 各畳み込み層の出力サイズを手計算で確認
   - フィルタ数の増加（32→64→64）の意味を理解
   - プーリング層による次元削減の効果を学習

3. **過学習対策の実践** (2-3時間)
   - Dropoutの効果を実験（0.3, 0.5, 0.7で比較）
   - Early Stoppingのpatience値を変更して効果を確認
   - 学習率スケジューリングの動作を観察

4. **発展実験** (4-6時間)
   - モデル構造を変更（層数、フィルタ数）
   - 異なる活性化関数（ReLU、LeakyReLU）を試行
   - バッチ正規化の追加実験

**習得スキル:**
- ✅ CNNアーキテクチャ設計の基本
- ✅ 過学習対策の実践的手法
- ✅ ハイパーパラメータチューニング
- ✅ モデル改良の思考プロセス

---

### 🚀 **レベル3: 実用技術（推奨学習期間: 3-4週間）**

#### ステップ3: 転移学習
**目標**: 業務レベルの効率的な学習手法を習得

**学習手順:**
1. **基本実行** (1時間)
   ```bash
   python advanced/transfer_learning.py
   ```
   - デモを実行して転移学習の効果を体感

2. **モデル比較実験** (4-6時間)
   - VGG16、ResNet50、MobileNetV2の性能比較
   - 学習時間、精度、モデルサイズの違いを記録
   - 各モデルの特徴と適用場面を理解

3. **2段階学習の理解** (3-4時間)
   - 固定学習とファインチューニングの効果の差を測定
   - 学習率の段階的調整の重要性を理解
   - 各段階での学習曲線の変化を観察

4. **データ拡張の効果測定** (2-3時間)
   - データ拡張有無での精度比較
   - 各種拡張手法の個別効果を検証
   - 小データセットでの転移学習の威力を確認

**習得スキル:**
- ✅ 事前学習済みモデルの効果的活用
- ✅ ファインチューニング戦略
- ✅ データ拡張の実装と評価
- ✅ 実用的な開発ワークフロー

---

### 🎯 **レベル4: 実践応用（推奨学習期間: 4-6週間）**

#### ステップ4: カスタムデータセット
**目標**: 独自のプロジェクトを実装できるスキル

**学習手順:**
1. **サンプル実行** (30分)
   ```bash
   python utils/custom_dataset.py
   ```
   - デモデータセットでの動作確認

2. **独自データセットの準備** (2-4時間)
   - インターネットから画像を収集（各クラス100枚以上）
   - 適切なディレクトリ構造で整理
   - データ品質の確認と改善

3. **データセット処理の実装** (4-6時間)
   - CustomImageDatasetクラスを独自データに適用
   - クラス不均衡の対処
   - データ拡張パラメータの最適化

4. **エンドツーエンドモデル構築** (6-10時間)
   - カスタムデータセットに転移学習を適用
   - モデル性能の評価と改善
   - 本格的な画像認識アプリケーションの完成

**習得スキル:**
- ✅ 実際のプロジェクトでのデータ準備
- ✅ データ品質管理
- ✅ エンドツーエンドの開発能力
- ✅ 実業務レベルの問題解決力

---

### 💡 **学習のコツとベストプラクティス**

#### 効率的な学習方法
1. **手を動かす**: 必ずコードを実行して結果を確認
2. **比較実験**: パラメータを変更して違いを体感
3. **可視化活用**: 学習曲線、混同行列を必ず確認
4. **記録をつける**: 実験結果をノートやスプレッドシートに記録

#### つまずきやすいポイント
- **環境構築**: 最初のライブラリインストールで躓く場合が多い
- **計算時間**: GPUなしの場合は学習に時間がかかることを理解
- **過学習の判断**: 学習曲線の読み方に慣れるまで時間がかかる
- **ハイパーパラメータ**: 一度に多くを変更せず、一つずつ実験

#### 次のステップへの発展
- **Kaggleコンペティション**: 実際のコンペで腕試し
- **専門分野への応用**: 医療、農業、製造業など興味のある分野
- **最新研究の追跡**: arXiv、学会論文で最新トレンドを学習
- **実装力向上**: PyTorchなど他のフレームワークにも挑戦

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