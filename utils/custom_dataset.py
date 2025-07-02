#!/usr/bin/env python3
"""
カスタムデータセット用のユーティリティクラス
独自の画像データセットを使用した機械学習を支援

対応する機能:
- フォルダ構造からの自動データセット作成
- データ拡張 (Data Augmentation)
- 学習/検証/テスト分割
- バッチ処理
- 画像の前処理とリサイズ
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

class CustomImageDataset:
    def __init__(self, data_dir, image_size=(224, 224), batch_size=32):
        """
        カスタムデータセットの初期化
        
        Args:
            data_dir (str): データディレクトリのパス
            image_size (tuple): 画像のリサイズサイズ
            batch_size (int): バッチサイズ
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.classes = []
        self.class_to_idx = {}
        self.images = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
    def scan_directory(self):
        """
        ディレクトリをスキャンして画像とラベルを収集
        期待されるディレクトリ構造:
        data_dir/
        ├── class1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class2/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── ...
        """
        print(f"データディレクトリをスキャン中: {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {self.data_dir}")
        
        # サポートされる画像形式
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # クラスディレクトリの取得
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.classes = sorted([d.name for d in class_dirs])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        print(f"発見されたクラス数: {len(self.classes)}")
        print(f"クラス: {self.classes}")
        
        # 各クラスの画像を収集
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            # クラスディレクトリ内の画像ファイルを取得
            image_files = [
                f for f in class_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_formats
            ]
            
            print(f"クラス '{class_name}': {len(image_files)}枚の画像")
            
            for image_file in image_files:
                self.images.append(str(image_file))
                self.labels.append(class_idx)
        
        print(f"総画像数: {len(self.images)}")
        return len(self.images)
    
    def load_and_preprocess_image(self, image_path):
        """画像の読み込みと前処理"""
        try:
            # 画像の読み込み
            image = Image.open(image_path)
            
            # RGBに変換（必要に応じて）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # リサイズ
            image = image.resize(self.image_size)
            
            # NumPy配列に変換
            image_array = np.array(image, dtype=np.float32)
            
            # 正規化 (0-255 → 0-1)
            image_array = image_array / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"画像読み込みエラー: {image_path}, エラー: {e}")
            # エラー時はゼロ配列を返す
            return np.zeros((*self.image_size, 3), dtype=np.float32)
    
    def create_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """データセットの作成と分割"""
        print("データセットを作成中...")
        
        if not self.images:
            self.scan_directory()
        
        # 画像の読み込み
        print("画像を読み込み中...")
        X = []
        for i, image_path in enumerate(self.images):
            if i % 100 == 0:
                print(f"進捗: {i}/{len(self.images)}")
            
            image_array = self.load_and_preprocess_image(image_path)
            X.append(image_array)
        
        X = np.array(X)
        y = np.array(self.labels)
        
        print(f"データセット形状: {X.shape}")
        print(f"ラベル形状: {y.shape}")
        
        # ラベルのワンホットエンコーディング
        y_categorical = keras.utils.to_categorical(y, len(self.classes))
        
        # データの分割
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"学習データ: {X_train.shape}")
        print(f"検証データ: {X_val.shape}")
        print(f"テストデータ: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_generators(self, validation_split=0.2):
        """データジェネレータの作成（データ拡張付き）"""
        print("データジェネレータを作成中...")
        
        # データ拡張設定
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # 検証用（データ拡張なし）
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # 学習用ジェネレータ
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # 検証用ジェネレータ
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # クラス情報の更新
        self.classes = list(train_generator.class_indices.keys())
        self.class_to_idx = train_generator.class_indices
        
        print(f"学習バッチ数: {len(train_generator)}")
        print(f"検証バッチ数: {len(val_generator)}")
        print(f"クラス: {self.classes}")
        
        return train_generator, val_generator
    
    def visualize_samples(self, data_generator=None, num_samples=9):
        """データサンプルの可視化"""
        if data_generator is None:
            # 直接ファイルから可視化
            if not self.images:
                self.scan_directory()
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.ravel()
            
            # ランダムにサンプルを選択
            indices = np.random.choice(len(self.images), num_samples, replace=False)
            
            for i, idx in enumerate(indices):
                image_path = self.images[idx]
                label = self.labels[idx]
                class_name = self.classes[label]
                
                image = self.load_and_preprocess_image(image_path)
                
                axes[i].imshow(image)
                axes[i].set_title(f'クラス: {class_name}')
                axes[i].axis('off')
            
        else:
            # ジェネレータから可視化
            batch_x, batch_y = next(data_generator)
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            axes = axes.ravel()
            
            for i in range(min(num_samples, len(batch_x))):
                image = batch_x[i]
                label_idx = np.argmax(batch_y[i])
                class_name = self.classes[label_idx]
                
                axes[i].imshow(image)
                axes[i].set_title(f'クラス: {class_name}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/datasets/custom_dataset_samples.png')
        plt.show()
    
    def get_class_distribution(self):
        """クラス分布の取得と可視化"""
        if not self.labels:
            self.scan_directory()
        
        unique, counts = np.unique(self.labels, return_counts=True)
        class_distribution = {self.classes[i]: counts[i] for i in unique}
        
        # 可視化
        plt.figure(figsize=(12, 6))
        plt.bar(class_distribution.keys(), class_distribution.values())
        plt.title('クラス分布')
        plt.xlabel('クラス')
        plt.ylabel('画像数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('/home/smitsu32/projects/machine-learning/image-recognition/datasets/class_distribution.png')
        plt.show()
        
        print("クラス分布:")
        for class_name, count in class_distribution.items():
            print(f"  {class_name}: {count}枚")
        
        return class_distribution
    
    def save_dataset_info(self, filepath=None):
        """データセット情報の保存"""
        if filepath is None:
            filepath = '/home/smitsu32/projects/machine-learning/image-recognition/datasets/dataset_info.json'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        dataset_info = {
            'data_dir': str(self.data_dir),
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'total_images': len(self.images),
            'class_distribution': self.get_class_distribution() if self.labels else {}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"データセット情報を保存しました: {filepath}")

def create_sample_dataset():
    """サンプルデータセットの作成"""
    print("サンプルデータセットを作成中...")
    
    sample_dir = Path('/home/smitsu32/projects/machine-learning/image-recognition/datasets/sample_data')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # サンプルクラスの作成
    classes = ['cat', 'dog', 'bird']
    
    for class_name in classes:
        class_dir = sample_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # 各クラスにサンプル画像を作成（ランダムなカラー画像）
        for i in range(5):
            # ランダムな色の画像を生成
            image = Image.fromarray(
                np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            )
            image.save(class_dir / f'{class_name}_{i:03d}.png')
    
    print(f"サンプルデータセットを作成しました: {sample_dir}")
    return str(sample_dir)

def main():
    """メイン実行関数"""
    print("カスタムデータセットユーティリティのデモを開始します...")
    
    # サンプルデータセットの作成
    sample_data_dir = create_sample_dataset()
    
    # データセットの作成
    dataset = CustomImageDataset(sample_data_dir, image_size=(224, 224), batch_size=8)
    
    # ディレクトリのスキャン
    dataset.scan_directory()
    
    # クラス分布の表示
    dataset.get_class_distribution()
    
    # データサンプルの可視化
    dataset.visualize_samples()
    
    # データジェネレータの作成
    train_gen, val_gen = dataset.create_data_generators(validation_split=0.3)
    
    # ジェネレータからのサンプル可視化
    print("データジェネレータからのサンプル:")
    dataset.visualize_samples(train_gen)
    
    # データセット情報の保存
    dataset.save_dataset_info()
    
    print("カスタムデータセットのデモが完了しました！")

if __name__ == "__main__":
    main()