#!/usr/bin/env python3
"""
MNIST手書き数字分類器
機械学習の入門に最適な画像認識コード

MNISTデータセット:
- 0-9の手書き数字画像
- 28x28ピクセルのグレースケール画像
- 60,000枚の学習データ、10,000枚のテストデータ
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
    
    def load_and_preprocess_data(self):
        """MNISTデータの読み込みと前処理"""
        print("MNISTデータセットを読み込み中...")
        
        # データセット読み込み
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # データの正規化 (0-255 → 0-1)
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # CNNのために次元を追加 (28, 28) → (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # ラベルをワンホットエンコーディング
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print(f"学習データ: {x_train.shape}, ラベル: {y_train.shape}")
        print(f"テストデータ: {x_test.shape}, ラベル: {y_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def create_simple_model(self):
        """シンプルな多層パーセプトロンモデル"""
        print("シンプルな多層パーセプトロンモデルを構築中...")
        
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def create_cnn_model(self):
        """畳み込みニューラルネットワークモデル"""
        print("CNNモデルを構築中...")
        
        model = keras.Sequential([
            # 第1畳み込み層
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # 第2畳み込み層
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # 全結合層
            layers.Flatten(),
            layers.Dropout(0.25),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
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
                factor=0.5,
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
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/basic/mnist_training_history.png')
        plt.show()
    
    def visualize_data_samples(self, x_data, y_data, num_samples=25):
        """データサンプルの可視化"""
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = x_data[i].reshape(28, 28)  # (28, 28, 1) → (28, 28)
            label = np.argmax(y_data[i])
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'ラベル: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/basic/mnist_data_samples.png')
        plt.show()
    
    def predict_and_visualize(self, x_test, y_test, num_samples=25):
        """予測結果の可視化"""
        predictions = self.model.predict(x_test[:num_samples])
        
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(num_samples):
            img = x_test[i].reshape(28, 28)
            true_label = np.argmax(y_test[i])
            pred_label = np.argmax(predictions[i])
            confidence = np.max(predictions[i])
            
            # 正解/不正解で色分け
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(
                f'実際: {true_label}, 予測: {pred_label}\n信頼度: {confidence:.2f}',
                color=color
            )
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/basic/mnist_predictions.png')
        plt.show()
    
    def create_confusion_matrix(self, x_test, y_test):
        """混同行列の作成と可視化"""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混同行列')
        plt.xlabel('予測ラベル')
        plt.ylabel('実際のラベル')
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/basic/mnist_confusion_matrix.png')
        plt.show()
    
    def save_model(self, filepath=None):
        """モデルの保存"""
        if filepath is None:
            filepath = '/home/smitsu32/projects/machine-learning/image-recognition/models/mnist_model.h5'
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"モデルを保存しました: {filepath}")

def main():
    """メイン実行関数"""
    print("MNIST手書き数字分類プログラムを開始します...")
    
    # 分類器のインスタンス作成
    classifier = MNISTClassifier()
    
    # データの読み込みと前処理
    (x_train, y_train), (x_test, y_test) = classifier.load_and_preprocess_data()
    
    # データサンプルの可視化
    print("データサンプルを可視化中...")
    classifier.visualize_data_samples(x_train, y_train)
    
    # モデル選択の選択肢を表示
    print("\nモデルを選択してください:")
    print("1. シンプルな多層パーセプトロン")
    print("2. 畳み込みニューラルネットワーク (推奨)")
    
    choice = input("選択 (1 or 2): ").strip()
    
    if choice == "1":
        classifier.create_simple_model()
        epochs = 15
    else:
        classifier.create_cnn_model()
        epochs = 12
    
    # モデルの学習
    classifier.train_model(x_train, y_train, x_test, y_test, epochs=epochs)
    
    # モデルの評価
    classifier.evaluate_model(x_test, y_test)
    
    # 学習履歴の可視化
    classifier.plot_training_history()
    
    # 予測結果の可視化
    classifier.predict_and_visualize(x_test, y_test)
    
    # 混同行列の作成
    classifier.create_confusion_matrix(x_test, y_test)
    
    # モデルの保存
    classifier.save_model()
    
    print("プログラムが完了しました！")

if __name__ == "__main__":
    main()