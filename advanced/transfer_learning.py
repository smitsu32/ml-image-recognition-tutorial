#!/usr/bin/env python3
"""
転移学習を使った画像分類器
事前学習済みモデル（VGG16, ResNet50, MobileNet等）を使用した効率的な画像分類

転移学習の利点:
- 少ないデータでも高い精度を達成
- 学習時間の短縮
- 計算リソースの節約
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class TransferLearningClassifier:
    def __init__(self, base_model_name='VGG16', num_classes=10):
        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.model = None
        self.base_model = None
        self.history = None
        
        # 利用可能な事前学習済みモデル
        self.available_models = {
            'VGG16': VGG16,
            'ResNet50': ResNet50,
            'MobileNetV2': MobileNetV2
        }
    
    def create_base_model(self, input_shape=(224, 224, 3)):
        """事前学習済みベースモデルの作成"""
        print(f"事前学習済み{self.base_model_name}モデルを読み込み中...")
        
        if self.base_model_name not in self.available_models:
            raise ValueError(f"サポートされていないモデル: {self.base_model_name}")
        
        base_model_class = self.available_models[self.base_model_name]
        
        # ImageNetで事前学習済みのモデルを読み込み（最上位層は除く）
        base_model = base_model_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # 最初は事前学習済みの重みを固定
        base_model.trainable = False
        
        print(f"ベースモデル: {base_model.name}")
        print(f"パラメータ数: {base_model.count_params():,}")
        
        self.base_model = base_model
        return base_model
    
    def create_transfer_model(self, input_shape=(224, 224, 3)):
        """転移学習モデルの構築"""
        print("転移学習モデルを構築中...")
        
        # ベースモデルの作成
        base_model = self.create_base_model(input_shape)
        
        # カスタムトップ層の追加
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # モデルのコンパイル
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        self.model = model
        return model
    
    def prepare_data_for_transfer_learning(self, x_data, target_size=(224, 224)):
        """転移学習用のデータ前処理"""
        print("転移学習用にデータを前処理中...")
        
        # 画像サイズの変更
        if len(x_data.shape) == 4:  # バッチデータの場合
            x_resized = tf.image.resize(x_data, target_size)
        else:
            x_resized = tf.image.resize(x_data[np.newaxis, ...], target_size)[0]
        
        # グレースケールの場合はRGBに変換
        if x_resized.shape[-1] == 1:
            x_resized = tf.image.grayscale_to_rgb(x_resized)
        
        # 事前学習済みモデル用の前処理
        if self.base_model_name == 'VGG16':
            x_preprocessed = keras.applications.vgg16.preprocess_input(x_resized)
        elif self.base_model_name == 'ResNet50':
            x_preprocessed = keras.applications.resnet50.preprocess_input(x_resized)
        elif self.base_model_name == 'MobileNetV2':
            x_preprocessed = keras.applications.mobilenet_v2.preprocess_input(x_resized)
        else:
            x_preprocessed = x_resized / 255.0
        
        return x_preprocessed.numpy()
    
    def train_transfer_model(self, x_train, y_train, x_val, y_val, 
                           epochs=20, batch_size=32, fine_tune=True):
        """転移学習モデルの学習"""
        print("転移学習を開始します...")
        
        # データ拡張
        data_augmentation = keras.Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        
        # 第1段階: ベースモデルを固定して学習
        print("第1段階: ベースモデル固定での学習")
        
        callbacks_stage1 = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # データ拡張を適用
        x_train_aug = data_augmentation(x_train)
        
        history_stage1 = self.model.fit(
            x_train_aug, y_train,
            batch_size=batch_size,
            epochs=epochs // 2,
            validation_data=(x_val, y_val),
            callbacks=callbacks_stage1,
            verbose=1
        )
        
        # 第2段階: ファインチューニング（オプション）
        if fine_tune:
            print("第2段階: ファインチューニング")
            
            # ベースモデルの上位層を学習可能にする
            self.base_model.trainable = True
            
            # 学習率を下げる
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            callbacks_stage2 = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=7,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-8
                )
            ]
            
            history_stage2 = self.model.fit(
                x_train_aug, y_train,
                batch_size=batch_size,
                epochs=epochs // 2,
                validation_data=(x_val, y_val),
                callbacks=callbacks_stage2,
                verbose=1
            )
            
            # 履歴の結合
            self.history = self.combine_histories(history_stage1, history_stage2)
        else:
            self.history = history_stage1
        
        return self.history
    
    def combine_histories(self, hist1, hist2):
        """2つの学習履歴を結合"""
        combined_history = {}
        for key in hist1.history.keys():
            combined_history[key] = hist1.history[key] + hist2.history[key]
        
        # keras.callbacks.History形式にラップ
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined_history)
    
    def evaluate_model(self, x_test, y_test):
        """モデルの評価"""
        print("モデルの評価中...")
        
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"テスト精度: {test_accuracy:.4f}")
        print(f"テスト損失: {test_loss:.4f}")
        
        # 詳細な分類レポート
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        print("\n分類レポート:")
        print(classification_report(y_true, y_pred))
        
        return test_accuracy, test_loss
    
    def plot_training_history(self):
        """学習履歴の可視化"""
        if self.history is None:
            print("学習履歴がありません")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 精度の推移
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title(f'{self.base_model_name} - モデル精度の推移')
        ax1.set_xlabel('エポック')
        ax1.set_ylabel('精度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 損失の推移
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title(f'{self.base_model_name} - モデル損失の推移')
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('損失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/home/smitsu32/projects/machine-learning/image-recognition/advanced/{self.base_model_name.lower()}_training_history.png')
        plt.show()
    
    def create_confusion_matrix(self, x_test, y_test, class_names=None):
        """混同行列の作成"""
        predictions = self.model.predict(x_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{self.base_model_name} - 混同行列')
        plt.xlabel('予測ラベル')
        plt.ylabel('実際のラベル')
        plt.tight_layout()
        plt.savefig(f'/home/smitsu32/projects/machine-learning/image-recognition/advanced/{self.base_model_name.lower()}_confusion_matrix.png')
        plt.show()
    
    def save_model(self, filepath=None):
        """モデルの保存"""
        if filepath is None:
            filepath = f'/home/smitsu32/projects/machine-learning/image-recognition/models/{self.base_model_name.lower()}_transfer_model.h5'
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"モデルを保存しました: {filepath}")

def demo_with_cifar10():
    """CIFAR-10データセットでの転移学習デモ"""
    print("CIFAR-10データセットでの転移学習デモを開始します...")
    
    # CIFAR-10データの読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # ラベルのワンホットエンコーディング
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # クラス名
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 転移学習分類器の作成
    classifier = TransferLearningClassifier(base_model_name='MobileNetV2', num_classes=10)
    
    # モデルの構築
    classifier.create_transfer_model(input_shape=(224, 224, 3))
    
    # データの前処理
    print("データの前処理中...")
    x_train_processed = classifier.prepare_data_for_transfer_learning(x_train)
    x_test_processed = classifier.prepare_data_for_transfer_learning(x_test)
    
    # 学習用データの一部を使用（デモ用）
    train_size = 5000
    x_train_demo = x_train_processed[:train_size]
    y_train_demo = y_train[:train_size]
    
    # モデルの学習
    classifier.train_transfer_model(
        x_train_demo, y_train_demo,
        x_test_processed, y_test,
        epochs=10, batch_size=32, fine_tune=True
    )
    
    # モデルの評価
    classifier.evaluate_model(x_test_processed, y_test)
    
    # 結果の可視化
    classifier.plot_training_history()
    classifier.create_confusion_matrix(x_test_processed, y_test, class_names)
    
    # モデルの保存
    classifier.save_model()
    
    print("転移学習デモが完了しました！")

def main():
    """メイン実行関数"""
    print("転移学習による画像分類プログラムを開始します...")
    
    print("デモを実行しますか？")
    print("1. CIFAR-10データセットでのデモ実行")
    print("2. カスタムデータでの実行（実装予定）")
    
    choice = input("選択 (1 or 2): ").strip()
    
    if choice == "1":
        demo_with_cifar10()
    else:
        print("カスタムデータでの実行は今後実装予定です。")
        print("現在はCIFAR-10デモのみ利用可能です。")

if __name__ == "__main__":
    main()