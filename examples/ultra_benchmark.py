#!/usr/bin/env python3
"""
🏛️ TEMPLE IAM - BENCHMARK ULTRA INSTINCT
Test de compression sur fichiers déjà optimisés
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import zstandard as zstd
import lzma
import gzip
import brotli
from tqdm import tqdm
sys.path.append('..')
from ultra_instinct_compression import UltraInstinctCompressor, DataType, CompressionLevel

class UltraBenchmark:
    """Benchmark Ultra Instinct sur fichiers déjà compressés"""
    
    def __init__(self):
        self.compressors = {
            'zstd': self._compress_zstd,
            'lzma': self._compress_lzma,
            'gzip': self._compress_gzip,
            'brotli': self._compress_brotli,
            'ultra_instinct': self._compress_ultra_instinct
        }
        
        self.file_types = {
            '.iso': 'Jeu vidéo/ISO',
            '.mkv': 'Vidéo 4K',
            '.mp4': 'Vidéo HD',
            '.jpg': 'Image JPEG',
            '.png': 'Image PNG',
            '.bin': 'Modèle IA',
            '.zip': 'Archive ZIP',
            '.rar': 'Archive RAR'
        }
        
        self.results = []
        plt.style.use('dark_background')
    
    def _compress_zstd(self, data: bytes) -> Tuple[bytes, float]:
        """Test avec zstd (niveau max)"""
        start = time.perf_counter()
        cctx = zstd.ZstdCompressor(level=22)
        compressed = cctx.compress(data)
        return compressed, time.perf_counter() - start
    
    def _compress_lzma(self, data: bytes) -> Tuple[bytes, float]:
        """Test avec LZMA (niveau max)"""
        start = time.perf_counter()
        compressed = lzma.compress(data, preset=9)
        return compressed, time.perf_counter() - start
    
    def _compress_gzip(self, data: bytes) -> Tuple[bytes, float]:
        """Test avec gzip (niveau max)"""
        start = time.perf_counter()
        compressed = gzip.compress(data, compresslevel=9)
        return compressed, time.perf_counter() - start
    
    def _compress_brotli(self, data: bytes) -> Tuple[bytes, float]:
        """Test avec Brotli (niveau max)"""
        start = time.perf_counter()
        compressed = brotli.compress(data, quality=11)
        return compressed, time.perf_counter() - start
    
    def _compress_ultra_instinct(self, data: bytes) -> Tuple[bytes, float]:
        """Test avec notre compresseur Ultra Instinct"""
        start = time.perf_counter()
        compressor = UltraInstinctCompressor()
        compressed, metrics = compressor.compress_demo(
            np.frombuffer(data, dtype=np.uint8),
            DataType.BINARY
        )
        return compressed.tobytes(), time.perf_counter() - start
    
    def benchmark_file(self, file_path: str):
        """Lance le benchmark sur un fichier"""
        print(f"\n🏛️ TEMPLE IAM - BENCHMARK ULTRA INSTINCT")
        print(f"⚡ Test de compression sur : {file_path}")
        
        # Détection type
        ext = Path(file_path).suffix.lower()
        file_type = self.file_types.get(ext, 'Fichier inconnu')
        print(f"📁 Type détecté : {file_type}")
        
        # Chargement
        print("\n📥 Chargement du fichier...")
        with open(file_path, 'rb') as f:
            data = f.read()
        original_size = len(data)
        print(f"📊 Taille originale : {original_size/1e9:.2f} Go")
        
        # Tests
        print("\n🔥 ACTIVATION MODE ULTRA INSTINCT")
        results = []
        for name, compress_func in self.compressors.items():
            print(f"\n⚡ Test avec {name}...")
            try:
                compressed, duration = compress_func(data)
                ratio = len(data) / len(compressed)
                gain = (1 - len(compressed)/len(data)) * 100
                results.append({
                    'name': name,
                    'original_size': original_size,
                    'compressed_size': len(compressed),
                    'ratio': ratio,
                    'gain': gain,
                    'duration': duration,
                    'speed': original_size / duration / 1e6  # MB/s
                })
                print(f"📊 Ratio: {ratio:.3f}x (gain: {gain:.1f}%)")
                print(f"⚡ Vitesse: {original_size/duration/1e6:.1f} MB/s")
            except Exception as e:
                print(f"❌ Erreur: {e}")
        
        self.results.extend(results)
        return results
    
    def plot_results(self, save_path: str = 'benchmark_results.png'):
        """Génère des graphiques impressionnants"""
        plt.figure(figsize=(15, 10))
        plt.suptitle("🏛️ TEMPLE IAM - BENCHMARK ULTRA INSTINCT ⚡", fontsize=16)
        
        # 1. Ratios de compression
        plt.subplot(221)
        names = [r['name'] for r in self.results]
        ratios = [r['ratio'] for r in self.results]
        plt.bar(names, ratios, color='cyan')
        plt.title("Ratios de Compression")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Gains en %
        plt.subplot(222)
        gains = [r['gain'] for r in self.results]
        plt.bar(names, gains, color='magenta')
        plt.title("Gains (%)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Vitesses
        plt.subplot(223)
        speeds = [r['speed'] for r in self.results]
        plt.bar(names, speeds, color='yellow')
        plt.title("Vitesse (MB/s)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. Tailles
        plt.subplot(224)
        sizes_gb = [r['compressed_size']/1e9 for r in self.results]
        plt.bar(names, sizes_gb, color='green')
        plt.title("Taille Compressée (Go)")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

def demo_benchmark():
    """Démo du benchmark Ultra Instinct"""
    print("🏛️ TEMPLE IAM - BENCHMARK ULTRA INSTINCT")
    print("⚡ Version démo - Contactez-nous pour la version complète")
    
    # Demande fichier
    while True:
        file_path = input("\n📁 Chemin du fichier à tester (ex: jeu.iso, film.mkv) : ")
        if os.path.exists(file_path):
            break
        print("❌ Fichier non trouvé, réessayez...")
    
    # Lance benchmark
    benchmark = UltraBenchmark()
    results = benchmark.benchmark_file(file_path)
    
    # Affiche résultats
    print("\n=== RÉSULTATS ULTRA INSTINCT ===")
    best_ratio = max(r['ratio'] for r in results)
    best_speed = max(r['speed'] for r in results)
    
    print(f"\n🏆 MEILLEUR RATIO : {best_ratio:.3f}x")
    print(f"🚀 MEILLEURE VITESSE : {best_speed:.1f} MB/s")
    
    # Graphiques
    print("\n📊 GÉNÉRATION GRAPHIQUES...")
    benchmark.plot_results()
    
    print("\n⚡ ULTRA INSTINCT ACTIVÉ !")
    print("→ Version entreprise : compression 100x plus puissante")
    print("→ Contact : temple-iam.com")

if __name__ == "__main__":
    demo_benchmark() 