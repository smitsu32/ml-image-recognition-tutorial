#!/usr/bin/env python3
"""
CIFAR-10画像分類の基本的なCNNモデル
初心者向けの画像認識学習コード

CIFAR-10データセット:
- 10クラスの32x32カラー画像
- 各クラス6000枚、計60000枚の学習データ
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# GPU使用時のメモリ制限設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU設定エラー: {e}")

class CIFAR10Classifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
    def load_and_preprocess_data(self):
        """CIFAR-10データの読み込みと前処理"""
        print("CIFAR-10データセットを読み込み中...")
        
        # データセット読み込み
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # データの正規化 (0-255 → 0-1)
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # ラベルをワンホットエンコーディング
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"学習データ: {x_train.shape}, ラベル: {y_train.shape}")
        print(f"テストデータ: {x_test.shape}, ラベル: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def create_model(self):
        """基本的なCNNモデルの構築"""
        print("CNNモデルを構築中...")
        
        model = keras.Sequential([
            # 第1畳み込み層
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # 第2畳み込み層
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # 第3畳み込み層
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # 全結合層への展開
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),  # 過学習防止
            layers.Dense(10, activation='softmax')  # 10クラス分類
        ])
        
        # モデルのコンパイル
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # モデル構造の表示
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        """モデルの学習"""
        print(f"モデルの学習を開始します... (エポック数: {epochs})")
        
        # コールバック設定
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        # 学習実行
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """モデルの評価"""
        print("モデルの評価中...")
        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"テスト精度: {test_accuracy:.4f}")
        print(f"テスト損失: {test_loss:.4f}")
        
        return test_accuracy, test_loss
    
    def plot_training_history(self):
        """学習履歴の可視化"""
        if self.history is None:
            print("学習履歴がありません")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 精度の推移
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('モデル精度の推移')
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('精度')
        ax1.legend()
        ax1.grid(True)
        
        # 損失の推移
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('モデル損失の推移')
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('損失')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/basic/cifar10_training_history.png')
        plt.show()
        
    def predict_and_visualize(self, x_test, y_test, num_samples=9):
        """予測結果の可視化"""
        predictions = self.model.predict(x_test[:num_samples])
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # 画像表示
            axes[i].imshow(x_test[i])
            axes[i].set_title(
                f'実際: {self.class_names[np.argmax(y_test[i])]}\n'
                f'予測: {self.class_names[np.argmax(predictions[i])]}\n'
                f'信頼度: {np.max(predictions[i]):.2f}'
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/basic/cifar10_predictions.png')
        plt.show()
    
    def save_model(self, filepath=None):
        """モデルの保存"""
        if filepath is None:
            filepath = '/home/smitsu32/projects/machine-learning/image-recognition/models/cifar10_cnn_model.h5'
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"モデルを保存しました: {filepath}")

def main():
    """メイン実行関数"""
    print("CIFAR-10画像分類プログラムを開始します...")
    
    # 分類器のインスタンス作成
    classifier = CIFAR10Classifier()
    
    # データの読み込みと前処理
    (x_train, y_train), (x_test, y_test) = classifier.load_and_preprocess_data()
    
    # モデルの構築
    classifier.create_model()
    
    # モデルの学習
    classifier.train_model(x_train, y_train, x_test, y_test, epochs=20)
    
    # モデルの評価
    classifier.evaluate_model(x_test, y_test)
    
    # 学習履歴の可視化
    classifier.plot_training_history()
    
    # 予測結果の可視化
    classifier.predict_and_visualize(x_test, y_test)
    
    # モデルの保存
    classifier.save_model()
    
    print("プログラムが完了しました！")

if __name__ == "__main__":
    main()