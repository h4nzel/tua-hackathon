import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

# Import moon-dl modules
from importlib.machinery import SourceFileLoader
moon = SourceFileLoader("moon_dl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "moon-dl.py")).load_module()

def run_benchmark(num_tests=30):
    GRID_BOYUTU = 100
    HEIGHTMAP_DOSYA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_moonfesatan.png")
    YOLO_TEST_GORSEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "images", "test3.png")
    
    # ── 1. Arazi Verisi Yükleme ──
    print("Mekansal Veri yükleniyor...")
    if os.path.exists(HEIGHTMAP_DOSYA):
        yukleyici = moon.AyHeightmapYukleyici(dosya_yolu=HEIGHTMAP_DOSYA, grid_boyutu=GRID_BOYUTU)
    else:
        yukleyici = moon.AyAraziUretici(grid_boyutu=GRID_BOYUTU)
    heightmap, roughness_map, crater_rim_map = yukleyici.yukle() if hasattr(yukleyici, 'yukle') else yukleyici.arazi_uret()
    
    # ── 2. YOLO & PyCBC (Hızlı mocklama) ──
    # Normalde bu adım uzun sürer, ama bir kere çalıştıracağımız için:
    krater_haritasi = np.zeros((GRID_BOYUTU, GRID_BOYUTU), dtype=bool)
    tehlike_haritasi = np.zeros((GRID_BOYUTU, GRID_BOYUTU))
    if os.path.exists(YOLO_TEST_GORSEL):
        krater_haritasi = moon.KraterTespitci(grid_boyutu=GRID_BOYUTU).tespit_et(YOLO_TEST_GORSEL)
    tehlike_haritasi = moon.RoverDopplerAnalizci(grid_boyutu=GRID_BOYUTU).tehlike_haritasi_olustur(heightmap, roughness_map, crater_rim_map)
    
    # ── 3. Graf Hazırlama ──
    ay_yuzeyi = moon.AyYuzeyi(grid_boyutu=GRID_BOYUTU)
    ay_yuzeyi.graf_olustur(heightmap, roughness_map, crater_rim_map)
    ay_yuzeyi.set_krater_haritasi(krater_haritasi)
    ay_yuzeyi.set_tehlike_haritasi(tehlike_haritasi)
    ay_yuzeyi.maliyetleri_guncelle(0.0)
    
    dlalt = moon.AyDLALT(ay_yuzeyi)
    dlalt.landmark_seciciyi_egit(epochs=30)
    dlalt.landmarklari_sec(k=5)
    
    # ── 4. Benchmark Döngüsü ──
    dijkstra_nodes = []
    dijkstra_times = []
    dlalt_nodes = []
    dlalt_times = []
    
    print(f"\n--- Benchmark Başlıyor ({num_tests} Rota) ---")
    random.seed(42)
    
    basarili_rota = 0
    while basarili_rota < num_tests:
        x1, y1 = random.randint(5, 95), random.randint(5, 95)
        x2, y2 = random.randint(5, 95), random.randint(5, 95)
        
        # Çok yakın olmasınlar (en az 40 birim hücre)
        if (x1-x2)**2 + (y1-y2)**2 < 40**2:
            continue
            
        b_id = ay_yuzeyi._dugum_id(x1, y1)
        h_id = ay_yuzeyi._dugum_id(x2, y2)
        
        # Node tehlikeli bir yerde mi kontrol et:
        if ay_yuzeyi.dugumler[b_id].krater_no_go or ay_yuzeyi.dugumler[h_id].krater_no_go:
            continue
            
        print(f"Rota {basarili_rota+1}/{num_tests}: {b_id} -> {h_id}")
        
        # Dijkstra (Standart Yaklaşım)
        t0 = time.perf_counter()
        res_dijkstra = ay_yuzeyi.dijkstra_rota_bul(b_id, h_id)
        t_dij = (time.perf_counter() - t0) * 1000
        n_dij = getattr(ay_yuzeyi, 'son_ziyaret_edilen', 0)
        
        # DL-ALT
        t0 = time.perf_counter()
        res_dlalt = dlalt.dl_alt_rota_bul(b_id, h_id)
        t_dl = (time.perf_counter() - t0) * 1000
        n_dl = getattr(dlalt, 'son_ziyaret_edilen', 0)
        
        if res_dijkstra and res_dlalt:
            print(f"  Dijkstra: {n_dij} düğüm, {t_dij:.1f}ms")
            print(f"  DL-ALT:   {n_dl} düğüm,   {t_dl:.1f}ms")
            
            dijkstra_nodes.append(n_dij)
            dijkstra_times.append(t_dij)
            dlalt_nodes.append(n_dl)
            dlalt_times.append(t_dl)
            basarili_rota += 1

    # --- 5. Grafiği Çiz ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    algorithms = ['Standart Dijkstra', 'DL-ALT']
    avg_dij_n, avg_dl_n = np.median(dijkstra_nodes), np.median(dlalt_nodes)
    avg_dij_t, avg_dl_t = np.median(dijkstra_times), np.median(dlalt_times)
    
    nodes_explored = [avg_dij_n, avg_dl_n]
    time_ms = [avg_dij_t, avg_dl_t]
    colors = ['#f85149', '#3fb950']

    # Plot 1: Nodes Explored
    bars1 = ax1.bar(algorithms, nodes_explored, color=colors, width=0.5)
    ax1.set_title('Araştırılan Düğüm Medyan Sayısı)', color='#e6edf3', pad=15, fontweight='bold')
    ax1.set_ylabel('Düğüm Sayısı', color='#8b949e')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(nodes_explored)*0.02,
                 f'{int(height)}', ha='center', va='bottom', color='#e6edf3', fontweight='bold')

    # Plot 2: Time Taken
    bars2 = ax2.bar(algorithms, time_ms, color=colors, width=0.5)
    ax2.set_title('Medyan Hesaplanma Süresi', color='#e6edf3', pad=15, fontweight='bold')
    ax2.set_ylabel('Süre (ms)', color='#8b949e')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(time_ms)*0.02,
                 f'{height:.1f}ms', ha='center', va='bottom', color='#e6edf3', fontweight='bold')

    plt.suptitle(f'Gerçek Performans Testi ({num_tests} Rastgele Rota Medyanı)\n(10.000 Düğümlü Karmaşık Ay Arazisi)', 
                 color='white', fontsize=14, fontweight='bold', y=1.05)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n✅ Çıktı Grafiği Kaydedildi: {output_path}")

if __name__ == "__main__":
    run_benchmark()
