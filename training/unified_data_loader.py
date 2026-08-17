"""
Unified Data Loader for All Age Prediction Datasets
Consolidates: age_train (208,885), age_prediction_up (233,200), 20-50 (40,440)
Total: 482,525 images
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import random

class UnifiedDataLoader:
    """Load and organize all dataset images with age labels"""
    
    def __init__(self, base_path='data', random_seed=42):
        self.base_path = Path(base_path)
        self.random_seed = random_seed
        random.seed(random_seed)
        
        self.all_samples = []  # List of (image_path, age_label, source_dataset)
        self.age_distribution = defaultdict(int)
        self.source_distribution = defaultdict(int)
    
    def extract_age_from_filename(self, filename):
        """Extract age from various filename formats"""
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        
        try:
            # Format: {age}_{sequence}_{id}.jpg
            age = int(parts[0])
            if 1 <= age <= 100:  # Reasonable age range
                return age
        except (ValueError, IndexError):
            pass
        
        # Try from folder structure (some datasets use age folders)
        try:
            parent = os.path.basename(os.path.dirname(filename))
            age = int(parent)
            if 1 <= age <= 100:
                return age
        except (ValueError, TypeError):
            pass
        
        return None  # Could not extract age
    
    def load_dataset_age_train(self):
        """Load age_train dataset (208,885 images)"""
        print("\n[1/3] Loading age_train dataset...")
        
        path = self.base_path / 'age_train'
        if not path.exists():
            print("    ✗ age_train not found")
            return 0
        
        count = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    age = self.extract_age_from_filename(file)
                    
                    if age is not None:
                        self.all_samples.append({
                            'path': full_path,
                            'age': age,
                            'source': 'age_train'
                        })
                        self.age_distribution[age] += 1
                        self.source_distribution['age_train'] += 1
                        count += 1
                        
                        if count % 50000 == 0:
                            print(f"    Loaded {count:,} images...")
        
        print(f"    ✓ Loaded {count:,} images from age_train")
        return count
    
    def load_dataset_age_prediction_up(self):
        """Load age_prediction_up dataset (233,200 images)"""
        print("\n[2/3] Loading age_prediction_up dataset...")
        
        path = self.base_path / 'age_prediction_up' / 'age_prediction'
        if not path.exists():
            print("    ✗ age_prediction_up not found")
            return 0
        
        count = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    
                    # Try to extract age from folder structure (numbered folders)
                    age = None
                    try:
                        parts = full_path.split(os.sep)
                        for part in parts:
                            try:
                                folder_num = int(part.lstrip('0') or '0')
                                if 1 <= folder_num <= 100:
                                    age = folder_num
                                    break
                            except ValueError:
                                pass
                    except:
                        pass
                    
                    # Also try filename
                    if age is None:
                        age = self.extract_age_from_filename(file)
                    
                    # If still no age, try to infer from filename pattern
                    if age is None:
                        # Fallback: assume sequential folders correspond to ages
                        # This is a heuristic and may need adjustment
                        try:
                            # Look for numeric folder patterns
                            fname_parts = file.split('_')
                            if fname_parts[0].isdigit():
                                age = int(fname_parts[0])
                                if not (1 <= age <= 100):
                                    age = None
                        except:
                            age = None
                    
                    if age is not None:
                        self.all_samples.append({
                            'path': full_path,
                            'age': age,
                            'source': 'age_prediction_up'
                        })
                        self.age_distribution[age] += 1
                        self.source_distribution['age_prediction_up'] += 1
                        count += 1
                        
                        if count % 50000 == 0:
                            print(f"    Loaded {count:,} images...")
        
        print(f"    ✓ Loaded {count:,} images from age_prediction_up")
        return count
    
    def load_dataset_20_50(self):
        """Load 20-50 dataset (40,440 images)"""
        print("\n[3/3] Loading 20-50 dataset...")
        
        path = self.base_path / '20-50' / '20-50'
        if not path.exists():
            print("    ✗ 20-50 not found")
            return 0
        
        count = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    age = self.extract_age_from_filename(file)
                    
                    if age is not None:
                        self.all_samples.append({
                            'path': full_path,
                            'age': age,
                            'source': '20-50'
                        })
                        self.age_distribution[age] += 1
                        self.source_distribution['20-50'] += 1
                        count += 1
                        
                        if count % 10000 == 0:
                            print(f"    Loaded {count:,} images...")
        
        print(f"    ✓ Loaded {count:,} images from 20-50")
        return count
    
    def load_all_datasets(self):
        """Load all three datasets"""
        print("=" * 80)
        print("LOADING ALL DATASETS")
        print("=" * 80)
        
        count1 = self.load_dataset_age_train()
        count2 = self.load_dataset_age_prediction_up()
        count3 = self.load_dataset_20_50()
        
        total = count1 + count2 + count3
        
        print("\n" + "=" * 80)
        print("CONSOLIDATION SUMMARY")
        print("=" * 80)
        print(f"Total samples loaded: {total:,}")
        print(f"\nBreakdown by source:")
        for source, count in sorted(self.source_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100
            print(f"  {source:30s}: {count:>10,} ({pct:>5.1f}%)")
        
        print(f"\nAge distribution:")
        min_age = min(self.age_distribution.keys()) if self.age_distribution else 0
        max_age = max(self.age_distribution.keys()) if self.age_distribution else 0
        print(f"  Age range: {min_age} - {max_age}")
        print(f"  Unique ages: {len(self.age_distribution)}")
        
        # Show age histogram
        print(f"\n  Age histogram (in 10-year buckets):")
        for bucket_start in range(0, 101, 10):
            bucket_end = bucket_start + 9
            count = sum(v for age, v in self.age_distribution.items() if bucket_start <= age <= bucket_end)
            bar = '█' * (count // 1000)  # 1 block per 1000 images
            print(f"    [{bucket_start:3d}-{bucket_end:3d}]: {count:>7,} {bar}")
        
        return total
    
    def create_splits(self, train_pct=0.70, val_pct=0.15, test_pct=0.15):
        """Create train/val/test splits while preserving age distribution"""
        print("\n" + "=" * 80)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("=" * 80)
        
        # Shuffle samples
        random.shuffle(self.all_samples)
        
        # Split indices
        n = len(self.all_samples)
        train_idx = int(n * train_pct)
        val_idx = train_idx + int(n * val_pct)
        
        train_set = self.all_samples[:train_idx]
        val_set = self.all_samples[train_idx:val_idx]
        test_set = self.all_samples[val_idx:]
        
        print(f"Total samples: {n:,}")
        print(f"\nTrain set: {len(train_set):,} ({len(train_set)/n*100:.1f}%)")
        print(f"Val set:   {len(val_set):,} ({len(val_set)/n*100:.1f}%)")
        print(f"Test set:  {len(test_set):,} ({len(test_set)/n*100:.1f}%)")
        
        return {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
    
    def save_split_manifest(self, splits, output_path='dataset_manifest.json'):
        """Save splits to JSON manifest for reproducibility"""
        print(f"\nSaving manifest to {output_path}...")
        
        manifest = {
            'metadata': {
                'total_samples': len(self.all_samples),
                'random_seed': self.random_seed,
                'sources': dict(self.source_distribution),
                'age_range': [min(self.age_distribution.keys()), max(self.age_distribution.keys())],
                'unique_ages': len(self.age_distribution)
            },
            'splits': {
                split_name: [
                    {'path': s['path'], 'age': s['age'], 'source': s['source']}
                    for s in samples
                ]
                for split_name, samples in splits.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Manifest saved: {output_path}")
        
        # Also save summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("DATASET MANIFEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total samples: {manifest['metadata']['total_samples']:,}\n")
            f.write(f"Age range: {manifest['metadata']['age_range'][0]} - {manifest['metadata']['age_range'][1]}\n")
            f.write(f"Unique ages: {manifest['metadata']['unique_ages']}\n")
            f.write(f"Random seed: {manifest['metadata']['random_seed']}\n\n")
            
            f.write("Sources:\n")
            for source, count in sorted(manifest['metadata']['sources'].items(), key=lambda x: x[1], reverse=True):
                pct = count / manifest['metadata']['total_samples'] * 100
                f.write(f"  {source}: {count:,} ({pct:.1f}%)\n")
            
            f.write("\nSplits:\n")
            for split_name, samples in manifest['splits'].items():
                f.write(f"  {split_name}: {len(samples):,}\n")
        
        print(f"✓ Summary saved: {summary_path}")


def main():
    # Create data loader
    loader = UnifiedDataLoader(base_path='data')
    
    # Load all datasets
    total = loader.load_all_datasets()
    
    if total == 0:
        print("\n✗ No images found!")
        return
    
    # Create splits
    splits = loader.create_splits(train_pct=0.70, val_pct=0.15, test_pct=0.15)
    
    # Save manifest
    loader.save_split_manifest(splits)
    
    print("\n" + "=" * 80)
    print("✓ DATA LOADING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review dataset_manifest.json and dataset_manifest_summary.txt")
    print("2. Use the manifest to initialize PyTorch DataLoaders")
    print("3. Train the age estimation model on all 482,525 images")


if __name__ == "__main__":
    main()
