# Lunar Route Optimizer (AI-Powered Planetary Navigation System)

This project is an advanced, AI-powered navigation system that enables planetary rovers to autonomously determine the **safest and most energy-efficient route** across complex topographies, avoiding dangerous surface conditions (such as regolith sinkage and slippage) and physical obstacles like lunar craters. 

By integrating **DL-ALT**, **YOLO**, and **PyCBC**, the system unifies state-of-the-art approaches in autonomous driving, computer vision, and terrain signal processing.

---

## 🚀 Key Features and Core Modalities

The architecture fuses three primary artificial intelligence and analysis modules, unifying them under a single comprehensive mathematical cost function:

1. **GNN-Based DL-ALT Algorithm (Speed and Efficiency Scaling):**
   Unlike traditional Dijkstra or A* algorithms, this project utilizes Graph Neural Networks (GNN) to learn and place "strategic navigation waypoints" (Landmarks) across the landscape. This drastically prunes the search space.
2. **Computer Vision-Based Crater Detection with YOLO (Physical Safety):** 
   High-resolution orbital imagery (or drone telemetry) is digested by an ONNX-optimized YOLO model. Craters of various scales are detected and flagged as "No-Go Zones" on the navigation grid.
3. **Doppler / Signal Anomaly Detection with PyCBC (Terrain Security):**
   By simulating rover wheel spin frequencies (leveraging Doppler shift principles) and analyzing them via PyCBC's `matched_filter` algorithm, the system calculates the probability of the rover sinking into soft regolith or slipping near crater rims, generating a dynamic "Hazard Map."

---

## 🧠 Anatomy of the Cost Function and Graph Representation

The system models the lunar surface as a $100 \times 100$ graph ($10,000$ nodes, $\sim78,000$ edges). The traversal cost between two adjacent nodes ($u, v$) is determined by a multi-factorial equation:

$$ \text{Cost}(u, v) = \text{Distance} \times \text{Slope}_{\text{factor}} \times \text{Energy}_{\text{factor}} \times \text{Hazard}_{\text{factor}} \times \text{Crater}_{\text{penalty}} $$

- **Distance:** The 3D Euclidean distance between coordinates.
- **Slope:** Derived from the elevation gradient. Steep inclines result in high energy consumption penalties.
- **Energy (Shadow and Roughness):** Captures localized variance (roughness) and time-dependent solar illumination (shadowed areas are critical hazards for solar-powered rovers).
- **Hazard (PyCBC):** The anomaly score derived from signal processing, representing the risk of getting stuck.
- **Crater Penalty (YOLO):** If a node falls within a YOLO-detected crater bounding box, the cost is uniformly amplified by **$50\times$** (acting as a strict hard-wall constraint).

---

## ⚡ Speed and Efficiency (The DL-ALT Advantage)

The **DL-ALT (Deep Learning with A*, Landmarks, and Triangle inequality)** algorithm contributes to the mission software in the following ways:

1. **Bottleneck Prevention:** Classical A* algorithms can easily get trapped in local minima (dead ends) while navigating around massive craters, scanning thousands of useless nodes. DL-ALT mitigates this.
2. **Landmark Utilization:** The graph is indexed against $k=8$ critical "Landmarks" strategically identified by a trained Graph Convolutional Network (GCN).
3. **Triangle Inequality Heuristic:** 
   The distance from node $u$ to target $t$ can be safely lower-bounded via a Landmark $L$ using the rule: 
   $$ h(u, t) = \max_L | \text{dist}(u, L) - \text{dist}(t, L) | $$
4. **Benchmark Results (Empirical Testing):** Across 30 randomized lunar terrain tests, standard Dijkstra averages ~5000-8000 node visited (nearly the entire map). DL-ALT, leveraging its GNN landmarks, strictly narrows the search space to ~800-2500 nodes. This yields a **60%-80% reduction in node exploration costs**. This is a vital capability for rovers with highly restricted RAM/CPU specifications operating in deep space.

**Comparative Performance Analysis:**
![Performance Benchmark](file:///Users/flaner/Projects/tua-ml/performance_comparison.png)

---

## 🤖 Gemini Chat Assistant Integration
We have upgraded the backend into an autonomous Multi-Agent system via **Google Gemini**. By enabling native **Function Calling**, the LLM holds direct conversational authority over the DL-ALT algorithms. 

You can dispatch natural language commands to the rover (e.g., *"Chart a safe route from coordinate 10,10 to 90,90 avoiding craters"*). Gemini will orchestrate the YOLO, PyCBC, and Pathfinding services autonomously, synthesize the final routing array, construct the 8-panel dashboard, and formulate a conversational payload back to the client natively. 

*(Note: Requires a valid `GEMINI_API_KEY` defined in the environment).*

---

## 🛠️ Installation and Execution

### Requirements
The project requires a modern Python ($3.11+$) environment. You can manage dependencies using `uv` (recommended) or `pip`.

```bash
uv pip install torch torchvision torchaudio
uv pip install torch_geometric numpy scipy matplotlib pillow
uv pip install pycbc ultralytics opencv-python onnxruntime
uv pip install fastapi uvicorn pydantic pydantic-settings google-generativeai python-multipart
```

### Starting the Server (FastAPI)

The project has been refactored into a scalable Layered/Microservices REST-API architecture. To boot the system:

```bash
uv run uvicorn src.main:app --reload
```

The server will initialize at `http://127.0.0.1:8000`. You can test endpoints via the interactive Swagger UI at `http://127.0.0.1:8000/docs`.

### Dispatching Requests via CLI (cURL)

To manually trigger a route optimization payload, submit a POST request:

```bash
# JSON Interface
curl -X POST "http://127.0.0.1:8000/api/v1/routes/calculate" -H "Content-Type: application/json" -d '{
    "start": {"x": 10, "y": 10},
    "target": {"x": 90, "y": 90},
    "time_offset_hours": 0.0,
    "algorithm": "dl-alt"
}'

# Gemini Natural Language Chat Interface
curl -X POST "http://127.0.0.1:8000/api/v1/chat/message" \
     -H "Content-Type: multipart/form-data" \
     -F "prompt=Hello! Can you draw a safe route from (10,10) to (90,90) for me?"
```

The internal lifecycle processes as follows:
1. Captured by `route_controller.py`.
2. `terrain_service.py` extracts telemetry maps.
3. `crater_detection_service.py` fires YOLO inference.
4. `hazard_detection_service.py` runs PyCBC signal profiling.
5. `route_optimizer_service.py` calculates the optimal path using DL-ALT.
6. Returns an `HTTP 200 OK` JSON wrapper containing traversal telemetry, while independently rendering a comprehensive `lunar_route_result.png` multi-panel visual in the root directory.

---

## 📊 Visual Output Panels

Upon successful optimization, the system generates an 8-panel data visual encompassing:

1. **Elevation:** Colored topological map overlaying detected obstacles.
2. **Illumination:** Time-dependent dynamic shadowing.
3. **Slope:** Gradient-based climbing difficulty profiling.
4. **Roughness:** Standard deviation variance mapping of localized elevations.
5. **PyCBC Hazard Map:** Thermal heatmap representing sinkage and slippage anomaly flags.
6. **YOLO Crater Detection:** Monochrome exclusion zone mapping bounded by ONNX inferences.
7. **3D Substrate Plot:** `matplotlib.plot_surface` isometric rendering of the topographic terrain and optimal path.
8. **Original Telemetry:** The raw satellite payload merged with the final projected vector.

Equipped with this SOTA multi-modal stack, any integrated Lunar Rover will bypass craters, navigate out of topological dead ends, and evade soft regolith traps, securing a completely risk-free and energy-optimized traversal schema.
