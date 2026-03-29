import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from src.core.graph import LunarSurfaceGraph
from src.core.config import settings
from PIL import Image

logger = logging.getLogger("LunarRouter.Visualizer")

class LunarVisualizer:
    @staticmethod
    def render(
        graph: LunarSurfaceGraph,
        route_dlalt: list = None,
        route_dijkstra: list = None,
        output_file: str = "lunar_route_result.png",
        title: str = "Lunar Surface Routing Engine (PyCBC + YOLO + DL-ALT)"
    ):
        logger.info(f"Rendering visualization to {output_file}...")
        
        n = settings.GRID_SIZE
        h_map = np.zeros((n, n))
        i_map = np.zeros((n, n))
        s_map = np.zeros((n, n))
        r_map = np.zeros((n, n))
        hz_map = np.zeros((n, n))
        c_map = np.zeros((n, n))
        
        for y in range(n):
            for x in range(n):
                node = graph.nodes[graph.get_node_id(x, y)]
                h_map[y, x] = node.elevation
                i_map[y, x] = node.illumination
                s_map[y, x] = node.slope
                r_map[y, x] = node.roughness
                hz_map[y, x] = node.hazard_score
                c_map[y, x] = 1.0 if node.is_crater_no_go else 0.0

        orig_img = None
        if os.path.exists(settings.HEIGHTMAP_PATH):
            orig_img = np.array(Image.open(settings.HEIGHTMAP_PATH).convert('L'))
            
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(title, fontsize=20, fontweight='bold', color='white', y=0.98)
        fig.patch.set_facecolor('#1a1a1a')
        
        grid_axes = []
        if orig_img is not None:
            ax_orig = plt.subplot(2, 4, 1)
            ax_h = plt.subplot(2, 4, 2)
            ax_s = plt.subplot(2, 4, 3)
            ax_i = plt.subplot(2, 4, 4)
            ax_r = plt.subplot(2, 4, 5)
            ax_hz = plt.subplot(2, 4, 6)
            ax_c = plt.subplot(2, 4, 7)
            ax_3d = plt.subplot(2, 4, 8, projection='3d')
            grid_axes = [ax_orig, ax_h, ax_s, ax_i, ax_r, ax_hz, ax_c, ax_3d]
        else:
            ax_h = plt.subplot(3, 3, 1)
            ax_s = plt.subplot(3, 3, 2)
            ax_i = plt.subplot(3, 3, 3)
            ax_r = plt.subplot(3, 3, 4)
            ax_hz = plt.subplot(3, 3, 5)
            ax_c = plt.subplot(3, 3, 6)
            ax_3d = plt.subplot(3, 3, 8, projection='3d')
            grid_axes = [ax_h, ax_s, ax_i, ax_r, ax_hz, ax_c, ax_3d]

        for ax in grid_axes:
            if hasattr(ax, 'set_facecolor') and ax != ax_3d:
                ax.set_facecolor('#222')
            if hasattr(ax, 'tick_params'):
                ax.tick_params(colors='white')
            if hasattr(ax, 'title'):
                ax.title.set_color('white')
                
        def plot_routes(ax_layer, scale_x=1.0, scale_y=1.0):
            if route_dijkstra:
                rx, ry = [node.grid_x * scale_x for node in route_dijkstra], [node.grid_y * scale_y for node in route_dijkstra]
                ax_layer.plot(rx, ry, 'r--', linewidth=2, label='Dijkstra', alpha=0.6)
            if route_dlalt:
                rx, ry = [node.grid_x * scale_x for node in route_dlalt], [node.grid_y * scale_y for node in route_dlalt]
                ax_layer.plot(rx, ry, '#00ff00', linewidth=3, label='DL-ALT', alpha=0.9)
                ax_layer.plot(rx[0], ry[0], 'w*', markersize=15, label='Start')
                ax_layer.plot(rx[-1], ry[-1], 'y*', markersize=15, label='Target')
        
        # 1. Original
        if orig_img is not None:
            im0 = ax_orig.imshow(orig_img, cmap='gray')
            ax_orig.set_title("Orijinal Uydu Görseli (Real)")
            plt.colorbar(im0, ax=ax_orig, fraction=0.046, pad=0.04)
            sc_x = orig_img.shape[1] / n
            sc_y = orig_img.shape[0] / n
            plot_routes(ax_orig, sc_x, sc_y)

        # 2. Heightmap
        im1 = ax_h.imshow(h_map, cmap='terrain', origin='lower')
        ax_h.set_title("Topografi (İrtifa)")
        plt.colorbar(im1, ax=ax_h)
        plot_routes(ax_h)
        ax_h.legend(loc='upper right', facecolor='#222', labelcolor='white')

        # 3. Slope
        im2 = ax_s.imshow(s_map, cmap='inferno', origin='lower')
        ax_s.set_title("Eğim Haritası")
        plt.colorbar(im2, ax=ax_s)
        plot_routes(ax_s)

        # 4. Illumination
        im3 = ax_i.imshow(i_map, cmap='bone', origin='lower')
        ax_i.set_title(f"Gölge Analizi (Azimut: {int(graph.sun_azimuth)}°)")
        plt.colorbar(im3, ax=ax_i)
        plot_routes(ax_i)

        # 5. Roughness
        im4 = ax_r.imshow(r_map, cmap='magma', origin='lower')
        ax_r.set_title("Zemin Pürüzlülüğü")
        plt.colorbar(im4, ax=ax_r)
        plot_routes(ax_r)

        # 6. PyCBC Hazard
        im5 = ax_hz.imshow(hz_map, cmap='hot', origin='lower')
        ax_hz.set_title("PyCBC Doppler Tehlike")
        plt.colorbar(im5, ax=ax_hz)
        plot_routes(ax_hz)

        # 7. YOLO Crater No-Go Zone
        im6 = ax_c.imshow(c_map, cmap='Greys_r', origin='lower')
        ax_c.set_title("YOLO Krater No-Go Alanı")
        plot_routes(ax_c)

        # 8. 3D Plot
        X, Y = np.meshgrid(range(n), range(n))
        ax_3d.plot_surface(X, Y, h_map, cmap='terrain', alpha=0.7, edgecolor='none')
        ax_3d.set_title("3D Yüzey Modeli", pad=20)
        ax_3d.view_init(elev=45, azim=-45)
        ax_3d.set_axis_off()
        ax_3d.set_facecolor('#1a1a1a')
        
        if route_dlalt:
            z = [node.elevation + 20 for node in route_dlalt]
            rx, ry = [node.grid_x for node in route_dlalt], [node.grid_y for node in route_dlalt]
            ax_3d.plot(rx, ry, z, '#00ff00', linewidth=3)
        
        output_path = os.path.join(settings.BASE_DIR, output_file)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        logger.info(f"Visual dashboard saved successfully.")
        return output_path
