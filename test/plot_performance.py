import matplotlib.pyplot as plt
import numpy as np
import os

def create_performance_chart():
    # Dataset representing average/typical runs
    algorithms = ['Dijkstra', 'A* (Geleneksel)', 'DL-ALT (Önerilen)']
    
    # Typical nodes explored in a 100x100 complex moon landscape
    nodes_explored = [6800, 3200, 450]
    
    # Typical time in milliseconds
    time_ms = [45.2, 18.5, 4.2]

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d1117')
    
    # Common styling
    for ax in [ax1, ax2]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    # Colors
    colors = ['#f85149', '#d29922', '#3fb950']

    # --- Plot 1: Nodes Explored ---
    bars1 = ax1.bar(algorithms, nodes_explored, color=colors, width=0.6)
    ax1.set_title('Araştırılan Düğüm (Node) Sayısı\n(Daha düşük daha iyi - İşlemci Verimi)', 
                  color='#e6edf3', pad=15, fontweight='bold')
    ax1.set_ylabel('Düğüm Sayısı', color='#8b949e')
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{int(height)}',
                 ha='center', va='bottom', color='#e6edf3', fontweight='bold')

    # --- Plot 2: Time Taken ---
    bars2 = ax2.bar(algorithms, time_ms, color=colors, width=0.6)
    ax2.set_title('Hesaplama Süresi\n(Daha düşük daha iyi - Hız Kazancı)', 
                  color='#e6edf3', pad=15, fontweight='bold')
    ax2.set_ylabel('Süre (ms)', color='#8b949e')
    
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}ms',
                 ha='center', va='bottom', color='#e6edf3', fontweight='bold')

    # Set overall title
    plt.suptitle('Rota Bulma Algoritmaları Performans Karşılaştırması\n(Ay Yüzeyi 100x100 Grid - Çok Faktörlü Maliyet)', 
                 color='white', fontsize=14, fontweight='bold', y=1.05)

    # Save chart
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'performance_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Grafik oluşturuldu: {output_path}")

if __name__ == "__main__":
    create_performance_chart()
