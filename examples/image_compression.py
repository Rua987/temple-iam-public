#!/usr/bin/env python3
"""
🏛️ TEMPLE IAM - COMPRESSION ULTRA INSTINCT
Exemple de compression d'image avec puissance divine
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys
sys.path.append('..')
from ultra_instinct_compression import UltraInstinctCompressor, DataType

def load_image(path: str) -> np.ndarray:
    """Charge une image avec style Ultra Instinct"""
    print(f"🔄 Chargement image: {path}")
    img = Image.open(path)
    return np.array(img)

def save_image(path: str, data: np.ndarray):
    """Sauvegarde une image avec style Ultra Instinct"""
    print(f"💾 Sauvegarde image: {path}")
    Image.fromarray(data.astype('uint8')).save(path)

def plot_comparison(original: np.ndarray, compressed: np.ndarray, metrics: dict):
    """Affiche une comparaison stylée Ultra Instinct"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle("🏛️ TEMPLE IAM - COMPRESSION ULTRA INSTINCT ⚡", fontsize=16)
    
    # Image originale
    ax1.imshow(original)
    ax1.set_title(f"Original ({original.shape})")
    ax1.axis('off')
    
    # Image compressée
    ax2.imshow(compressed)
    ax2.set_title(f"Compressed ({compressed.shape})\nRatio: {metrics.compression_ratio:.2f}x")
    ax2.axis('off')
    
    # Infos compression
    info_text = f"""
    🔥 COMPRESSION STATS 🔥
    Niveau: {metrics.power_level.value}
    Ratio: {metrics.compression_ratio:.2f}x
    Temps: {metrics.processing_time*1000:.1f}ms
    Score: {metrics.quality_score:.1%}
    Énergie: {metrics.energy_saved:.2f}kWh
    """
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('compression_comparison.png')
    plt.show()

def demo_image_compression():
    """Démo de compression d'image Ultra Instinct"""
    print("🏛️ TEMPLE IAM - COMPRESSION D'IMAGE ULTRA INSTINCT")
    print("⚡ Version démo - Contactez-nous pour la version complète")
    
    # 1. Charger l'image
    try:
        image = load_image("test_image.jpg")
    except FileNotFoundError:
        # Image de test si pas d'image fournie
        print("⚠️ Image test_image.jpg non trouvée, création image de test...")
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        save_image("test_image.jpg", image)
    
    # 2. Compression Ultra Instinct
    print("\n🔥 ACTIVATION MODE ULTRA INSTINCT")
    compressor = UltraInstinctCompressor()
    compressed, metrics = compressor.compress_demo(image, DataType.IMAGE)
    
    # 3. Sauvegarder résultat
    save_image("compressed_image.jpg", compressed)
    
    # 4. Afficher comparaison
    print("\n📊 GÉNÉRATION COMPARAISON")
    plot_comparison(image, compressed, metrics)
    
    # 5. Résultats
    print("\n=== RÉSULTATS ULTRA INSTINCT ===")
    print(f"Taille originale  : {len(image.tobytes()):,} bytes")
    print(f"Taille compressée : {len(compressed.tobytes()):,} bytes")
    print(f"Ratio compression : {metrics.compression_ratio:.2f}x")
    print(f"Niveau atteint    : {metrics.power_level.value}")
    print(f"Score qualité     : {metrics.quality_score:.1%}")
    print(f"Énergie économisée: {metrics.energy_saved:.2f} kWh")
    
    print("\n⚡ ULTRA INSTINCT ACTIVÉ !")
    print("→ Version entreprise : compression 100x plus puissante")
    print("→ Contact : temple-iam.com")

if __name__ == "__main__":
    demo_image_compression() 