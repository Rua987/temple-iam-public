#!/usr/bin/env python3
"""
🏛️ TEMPLE IAM - COMPRESSION ULTRA INSTINCT
Exemple de compression vidéo avec puissance divine
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import sys
sys.path.append('..')
from ultra_instinct_compression import UltraInstinctCompressor, DataType

class VideoVisualizer:
    """Visualisation stylée Ultra Instinct pour la compression vidéo"""
    
    def __init__(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle("🏛️ TEMPLE IAM - COMPRESSION VIDÉO ULTRA INSTINCT ⚡", fontsize=16)
        
        # Setup des subplots
        self.ax1 = plt.subplot(221)  # Original
        self.ax2 = plt.subplot(222)  # Compressé
        self.ax3 = plt.subplot(212)  # Métriques
        
        # Style
        for ax in [self.ax1, self.ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Données de monitoring
        self.times = []
        self.ratios = []
        self.power_levels = []
        self.energy_saved = []
        
        # Ligne de monitoring
        self.line_ratio, = self.ax3.plot([], [], 'c-', label='Ratio', linewidth=2)
        self.line_energy, = self.ax3.plot([], [], 'g-', label='Énergie', linewidth=2)
        self.ax3.set_xlabel('Frame')
        self.ax3.set_ylabel('Valeur')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend()
        
        plt.tight_layout()
    
    def update(self, frame_num: int, original: np.ndarray, compressed: np.ndarray, metrics: dict):
        """Mise à jour de la visualisation"""
        # Images
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.imshow(original)
        self.ax2.imshow(compressed)
        self.ax1.set_title(f"Original (Frame {frame_num})")
        self.ax2.set_title(f"Compressed (Ratio: {metrics.compression_ratio:.2f}x)")
        
        # Métriques
        self.times.append(frame_num)
        self.ratios.append(metrics.compression_ratio)
        self.energy_saved.append(metrics.energy_saved)
        
        # Update des lignes
        self.line_ratio.set_data(self.times, self.ratios)
        self.line_energy.set_data(self.times, self.energy_saved)
        
        # Ajustement des axes
        self.ax3.relim()
        self.ax3.autoscale_view()
        
        # Info texte
        info_text = f"""
        🔥 COMPRESSION STATS 🔥
        Niveau: {metrics.power_level.value}
        Ratio: {metrics.compression_ratio:.2f}x
        Temps: {metrics.processing_time*1000:.1f}ms
        Score: {metrics.quality_score:.1%}
        Énergie: {metrics.energy_saved:.2f}kWh
        """
        if hasattr(self, 'text_box'):
            self.text_box.remove()
        self.text_box = self.fig.text(0.02, 0.02, info_text, fontsize=10,
                                    bbox=dict(facecolor='black', alpha=0.8))

def generate_test_video(frames: int = 30, size: tuple = (480, 640, 3)) -> np.ndarray:
    """Génère une vidéo de test avec motif Ultra Instinct"""
    print("🎬 Génération vidéo de test...")
    video = np.zeros((frames, *size))
    
    # Création d'un motif animé
    x = np.linspace(0, size[1]-1, size[1])
    y = np.linspace(0, size[0]-1, size[0])
    X, Y = np.meshgrid(x, y)
    
    for i in range(frames):
        # Motif animé style Ultra Instinct
        pattern = np.sin(X/30 + i/5) * np.cos(Y/30 + i/3)
        pattern = (pattern + 1) / 2  # Normalisation [0,1]
        
        # Ajout de couleurs
        video[i, :, :, 0] = pattern * np.sin(i/10)  # Rouge
        video[i, :, :, 1] = pattern * np.cos(i/8)   # Vert
        video[i, :, :, 2] = pattern * np.sin(i/6)   # Bleu
    
    return (video * 255).astype(np.uint8)

def demo_video_compression():
    """Démo de compression vidéo Ultra Instinct"""
    print("🏛️ TEMPLE IAM - COMPRESSION VIDÉO ULTRA INSTINCT")
    print("⚡ Version démo - Contactez-nous pour la version complète")
    
    # 1. Générer vidéo de test
    video = generate_test_video()
    print(f"📹 Vidéo générée: {video.shape}")
    
    # 2. Compression Ultra Instinct
    print("\n🔥 ACTIVATION MODE ULTRA INSTINCT")
    compressor = UltraInstinctCompressor()
    
    # 3. Visualisation temps réel
    visualizer = VideoVisualizer()
    
    def update(frame):
        # Compression de la frame
        original = video[frame]
        compressed, metrics = compressor.compress_demo(original, DataType.VIDEO)
        
        # Update visualisation
        visualizer.update(frame, original, compressed, metrics)
        
        return visualizer.fig,
    
    # Animation
    anim = FuncAnimation(visualizer.fig, update, frames=len(video),
                        interval=100, blit=True)
    plt.show()
    
    print("\n⚡ ULTRA INSTINCT ACTIVÉ !")
    print("→ Version entreprise : compression 100x plus puissante")
    print("→ Contact : temple-iam.com")

if __name__ == "__main__":
    demo_video_compression() 