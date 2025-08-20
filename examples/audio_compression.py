#!/usr/bin/env python3
"""
🏛️ TEMPLE IAM - COMPRESSION ULTRA INSTINCT
Exemple pur de compression audio style Karpathy
"""

import numpy as np
import sys
sys.path.append('..')
from ultra_instinct import UltraInstinctCompressor, DataType

def generate_test_audio(duration: float = 5.0, sample_rate: int = 44100) -> np.ndarray:
    """Pure : Génération d'un signal audio de test"""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Signal pur : combinaison de fréquences
    signal = (
        np.sin(2 * np.pi * 440 * t) * 0.5 +    # La (440 Hz)
        np.sin(2 * np.pi * 880 * t) * 0.3 +    # La octave sup
        np.sin(2 * np.pi * 1760 * t) * 0.2     # La 2 octaves sup
    )
    
    return (signal * 32767).astype(np.int16)

def load_audio(path: str) -> np.ndarray:
    """I/O : Chargement audio"""
    print(f"📥 Chargement audio: {path}")
    try:
        import soundfile as sf
        data, _ = sf.read(path)
        return (data * 32767).astype(np.int16)
    except:
        print("⚠️ Erreur de chargement, utilisation d'un signal de test")
        return generate_test_audio()

def save_audio(path: str, data: np.ndarray, sample_rate: int = 44100):
    """I/O : Sauvegarde audio"""
    print(f"💾 Sauvegarde audio: {path}")
    try:
        import soundfile as sf
        sf.write(path, data.astype(float) / 32767.0, sample_rate)
    except Exception as e:
        print(f"⚠️ Erreur de sauvegarde: {e}")

def print_metrics(metrics):
    """I/O : Affichage des métriques"""
    print("\n=== RÉSULTATS ULTRA INSTINCT ===")
    print(f"Taille originale  : {metrics.original_size:,} bytes")
    print(f"Taille compressée : {metrics.compressed_size:,} bytes")
    print(f"Ratio compression : {metrics.compression_ratio:.2f}x")
    print(f"Niveau atteint    : {metrics.power_level.value}")
    print(f"Score qualité     : {metrics.quality_score:.1%}")
    print(f"Énergie économisée: {metrics.energy_saved:.2f} kWh")

def demo_audio_compression():
    """Démo pure de compression audio"""
    print("🏛️ TEMPLE IAM - COMPRESSION AUDIO ULTRA INSTINCT")
    print("⚡ Version démo - Contactez-nous pour la version complète")
    
    # 1. I/O + Pure : Chargement ou génération
    try:
        audio = load_audio("test_audio.wav")
    except FileNotFoundError:
        print("⚠️ Fichier test_audio.wav non trouvé, génération signal de test...")
        audio = generate_test_audio()
        save_audio("test_audio.wav", audio)
    
    # 2. Pure : Compression
    print("\n🔥 ACTIVATION MODE ULTRA INSTINCT")
    compressor = UltraInstinctCompressor()
    compressed, metrics = compressor.compress_demo(audio, DataType.AUDIO)
    
    # 3. I/O : Sauvegarde
    save_audio("compressed_audio.wav", compressed)
    
    # 4. I/O : Résultats
    print_metrics(metrics)
    
    print("\n⚡ ULTRA INSTINCT ACTIVÉ !")
    print("→ Version entreprise : compression 100x plus puissante")
    print("→ Contact : temple-iam.com")

if __name__ == "__main__":
    demo_audio_compression() 