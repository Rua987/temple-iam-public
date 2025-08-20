#!/usr/bin/env python3
"""
🏛️ TEMPLE IAM - COMPRESSION ULTRA INSTINCT
Exemple pur de compression de texte style Karpathy
"""

import numpy as np
import sys
sys.path.append('..')
from ultra_instinct import UltraInstinctCompressor, DataType

def load_text(path: str) -> bytes:
    """I/O : Chargement du texte"""
    print(f"📥 Chargement texte: {path}")
    with open(path, 'rb') as f:
        return f.read()

def save_text(path: str, data: bytes):
    """I/O : Sauvegarde du texte"""
    print(f"💾 Sauvegarde texte: {path}")
    with open(path, 'wb') as f:
        f.write(data)

def print_metrics(metrics):
    """I/O : Affichage des métriques"""
    print("\n=== RÉSULTATS ULTRA INSTINCT ===")
    print(f"Taille originale  : {metrics.original_size:,} bytes")
    print(f"Taille compressée : {metrics.compressed_size:,} bytes")
    print(f"Ratio compression : {metrics.compression_ratio:.2f}x")
    print(f"Niveau atteint    : {metrics.power_level.value}")
    print(f"Score qualité     : {metrics.quality_score:.1%}")
    print(f"Énergie économisée: {metrics.energy_saved:.2f} kWh")

def demo_text_compression():
    """Démo pure de compression de texte"""
    print("🏛️ TEMPLE IAM - COMPRESSION DE TEXTE ULTRA INSTINCT")
    print("⚡ Version démo - Contactez-nous pour la version complète")
    
    # 1. I/O : Chargement
    try:
        text = load_text("test_text.txt")
    except FileNotFoundError:
        # Texte de test si pas de fichier fourni
        print("⚠️ Fichier test_text.txt non trouvé, création texte de test...")
        text = b"Temple IAM - Ultra Instinct " * 1000
        save_text("test_text.txt", text)
    
    # 2. Pure : Compression
    print("\n🔥 ACTIVATION MODE ULTRA INSTINCT")
    compressor = UltraInstinctCompressor()
    compressed, metrics = compressor.compress_demo(text, DataType.TEXT)
    
    # 3. I/O : Sauvegarde
    save_text("compressed_text.txt", compressed)
    
    # 4. I/O : Résultats
    print_metrics(metrics)
    
    print("\n⚡ ULTRA INSTINCT ACTIVÉ !")
    print("→ Version entreprise : compression 100x plus puissante")
    print("→ Contact : temple-iam.com")

if __name__ == "__main__":
    demo_text_compression() 