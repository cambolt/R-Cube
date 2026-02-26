import gradio as gr
import cv2
import numpy as np
import os
import time
import json
from cv_engine import extract_face, sample_colors_from_face, cluster_and_map_colors
from solver import map_colors_to_string, solve_cube

# Constants
try:
    with open("cube_3d.html", "r", encoding="utf-8") as f:
        CUBE_3D_HTML = f.read()
except:
    CUBE_3D_HTML = "<h3>3D Viewer Error: cube_3d.html not found</h3>"

SAFE_CUBE_HTML = CUBE_3D_HTML.replace('"', '&quot;').replace("'", '&apos;')

COLOR_VALUES = {
    'white': '#FFFFFF',
    'yellow': '#FFFF00',
    'red': '#FF0000',
    'orange': '#FFA500',
    'green': '#008000',
    'blue': '#0000FF'
}
COLOR_LIST = ['white', 'yellow', 'red', 'orange', 'green', 'blue']
FACE_NAMES = ['U', 'R', 'F', 'D', 'L', 'B']

# Initial solved state (54 colors)
INITIAL_STATE = []
for face in ['white', 'red', 'green', 'yellow', 'orange', 'blue']:
    INITIAL_STATE.extend([face] * 9)

# Kociemba standard string order: U1..U9, R1..R9, F1..F9, D1..D9, L1..L9, B1..B9

class RubiksApp:
    def __init__(self):
        self.logs = []
        self.clicks_a = []
        self.clicks_b = []

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        return "\n".join(self.logs[-15:]) # Keep last 15 lines

    def update_sticker(self, index, current_state):
        current_color = current_state[index]
        new_color_idx = (COLOR_LIST.index(current_color) + 1) % len(COLOR_LIST)
        new_color = COLOR_LIST[new_color_idx]
        current_state[index] = new_color
        
        # Return new state and update button class
        return current_state, gr.update(elem_classes=["sticker-btn", f"color-{new_color}"]), self.get_3d_update_js(current_state)

    def get_3d_update_js(self, state):
        state_dict = {}
        for i, face in enumerate(FACE_NAMES):
            state_dict[face] = state[i*9 : (i+1)*9]
        return f"window.updateCube3D({json.dumps(state_dict)});"

    # Updated Point Labels: Outer Hexagon 1-6 + Center 7
    PT_LABELS = [
        "1. 最上方顶点 (Rear-Top)", 
        "2. 左上方顶点 (West-Top)", 
        "3. 左下方顶点 (Front-Left-Bottom)", 
        "4. 最下方顶点 (South-Bottom)", 
        "5. 右下方顶点 (Front-Right-Bottom)", 
        "6. 右上方顶点 (East-Top)", 
        "7. 中心交汇处 (Front-Top corner / Center)"
    ]
    
    # Correct Cube Topology for drawing lines
    SKEL_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), # Outline hexagon
                  (6, 1), (6, 3), (6, 5)] # Edges from center to 2, 4, 6 (corners where faces meet)

    def handle_click_a(self, img, evt: gr.SelectData):
        if len(self.clicks_a) >= 7:
            return img, self.log("视角 A 已选满 (7/7)。如需重选请点击'重置状态'。")
        self.clicks_a.append((evt.index[0], evt.index[1]))
        return self.draw_and_guide(img, self.clicks_a, "A")

    def handle_click_b(self, img, evt: gr.SelectData):
        if len(self.clicks_b) >= 7:
            return img, self.log("视角 B 已选满 (7/7)。如需重选请点击'重置状态'。")
        self.clicks_b.append((evt.index[0], evt.index[1]))
        return self.draw_and_guide(img, self.clicks_b, "B")

    def draw_and_guide(self, img, clicks, view_name):
        count = len(clicks)
        img_out = img.copy()
        
        # Draw connections
        for start_idx, end_idx in self.SKEL_LINES:
            if count > start_idx and count > end_idx:
                cv2.line(img_out, clicks[start_idx], clicks[end_idx], (0, 255, 255), 2)

        # Draw points
        for i, p in enumerate(clicks):
            color = (0, 255, 0) if i == 6 else (255, 0, 100) # Green for center, Pinkish for others
            cv2.circle(img_out, p, 8, color, -1)
            cv2.putText(img_out, str(i+1), (p[0]+10, p[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if count < 7:
            msg = f"视角 {view_name}: 已点击 {count}/7。下一步: {self.PT_LABELS[count]}"
        else:
            if (view_name == "A" and len(self.clicks_b) < 7) or (view_name == "B" and len(self.clicks_a) < 7):
                msg = f"视角 {view_name} 已标完。请继续标注视角 {'B' if view_name == 'A' else 'A'}。"
            else:
                msg = f"视角 {view_name} 已标完。两个视角均已就绪，请点击'提取色彩识别'。"
            
        return img_out, self.log(msg)

    def clear_clicks(self):
        self.clicks_a = []
        self.clicks_b = []
        return None, None, self.log("点击坐标已清空")

    def identify_colors(self, img_a, img_b, current_state):
        if img_a is None or img_b is None:
            return current_state, self.log("错误: 请先上传两张图片"), "", None, None
        
        if len(self.clicks_a) < 7 or len(self.clicks_b) < 7:
            return current_state, self.log("提示: 每个视角需要点击 7 个关键点 (中心+6个角)"), "", None, None

        try:
            self.log("步骤 1/4: 正在对齐视角 A 采样点并提取颜色...")
            raw_colors_a = self.sample_colors_7pt(img_a, self.clicks_a, 0)
            
            self.log("步骤 2/4: 正在对齐视角 B 采样点并提取颜色...")
            raw_colors_b = self.sample_colors_7pt(img_b, self.clicks_b, 27)
            
            all_raw = raw_colors_a + raw_colors_b
            
            self.log("步骤 3/4: 正在进行颜色识别 (HSV + G/R ratio)...")
            # cv_engine returns colors in order: [U(0-8), F(9-17), R(18-26), D(27-35), B(36-44), L(45-53)]
            engine_state, _, _ = cluster_and_map_colors(all_raw)
            
            # Remap from cv_engine order [U,F,R,D,B,L] to app state order [U,R,F,D,L,B]
            # cv_engine: U=0-8,  F=9-17,  R=18-26, D=27-35, B=36-44, L=45-53
            # app state: U=0-8,  R=9-17,  F=18-26, D=27-35, L=36-44, B=45-53
            new_state = [None] * 54
            new_state[0:9]   = engine_state[0:9]    # U → U
            new_state[9:18]  = engine_state[18:27]   # R (engine pos 18-26) → app pos 9-17
            new_state[18:27] = engine_state[9:18]    # F (engine pos 9-17)  → app pos 18-26
            new_state[27:36] = engine_state[27:36]   # D → D
            new_state[36:45] = engine_state[45:54]   # L (engine pos 45-53) → app pos 36-44
            new_state[45:54] = engine_state[36:45]   # B (engine pos 36-44) → app pos 45-53
            
            self.log("步骤 4/4: 正在生成识别结果图像与 3D 同步...")
            # Draw result images (using engine_state order since draw uses start_idx 0/27 into engine order)
            res_a = self.draw_enhanced_results_7pt(img_a, self.clicks_a, engine_state, 0)
            res_b = self.draw_enhanced_results_7pt(img_b, self.clicks_b, engine_state, 27)
            
            # Generate UI updates for the 54 buttons (in ordered_btns order = app state order)
            btn_updates = []
            for color in new_state:
                btn_updates.append(gr.update(elem_classes=["sticker-btn", f"color-{color}"]))

            final_log = self.log("识别成功! 54个色块已识别, 3D 预览已更新。")
            return [new_state, final_log, self.get_3d_update_js(new_state), res_a, res_b] + btn_updates
        except Exception as e:
            import traceback
            traceback.print_exc()
            return current_state, self.log(f"识别异常: {str(e)}"), "", None, None

    def get_quads(self, pts):
        # pts[0]=1(top), pts[1]=2(NW), pts[2]=3(SW), pts[3]=4(bottom), pts[4]=5(SE), pts[5]=6(NE), pts[6]=7(center)
        # View A returns: [Top face, Left-side face, Right-side face]
        #   = [U, F, R] in physical cube terms
        # View B returns: [Top face, Left-side face, Right-side face]  
        #   = [D, B, L] in physical cube terms
        return [
            (pts[0], pts[5], pts[6], pts[1]), # Top/Bottom face
            (pts[1], pts[6], pts[3], pts[2]), # Left-side face (F for viewA, B for viewB)
            (pts[6], pts[5], pts[4], pts[3])  # Right-side face (R for viewA, L for viewB)
        ]


    def sample_colors_7pt(self, img, pts, start_idx):
        quads = self.get_quads(pts)
        face_colors = []
        for f_idx, quad in enumerate(quads):
            # Define a source square representing 3x3 units (0, 1, 2, 3)
            # The 4 corners of the 3x3 grid are (0,0), (3,0), (3,3), (0,3)
            src_pts = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
            dst_pts = np.float32(quad)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            for r in range(3):
                for c in range(3):
                    # Sample at centers: (0.5, 0.5), (1.5, 0.5), etc.
                    p = np.array([[[float(c) + 0.5, float(r) + 0.5]]], dtype=np.float32)
                    p_trans = cv2.perspectiveTransform(p, M)[0][0]
                    
                    if np.any(np.isnan(p_trans)):
                        self.log(f"警告: 采样点 ({r},{c}) 存在非数字 NaN")
                        face_colors.append((128, 128, 128))
                        continue

                    px, py = int(p_trans[0]), int(p_trans[1])
                    
                    # Ensure within bounds
                    px = max(2, min(img.shape[1]-3, px))
                    py = max(2, min(img.shape[0]-3, py))
                    
                    # Sample 5x5 area for noise reduction
                    sample = img[py-2:py+3, px-2:px+3]
                    if sample.size == 0:
                        face_colors.append((128, 128, 128))
                    else:
                        avg_col = np.mean(sample, axis=(0, 1))
                        # Check for NaN in avg_col
                        if np.any(np.isnan(avg_col)):
                             face_colors.append((128, 128, 128))
                        else:
                             face_colors.append(tuple(map(int, avg_col)))
        return face_colors

    def draw_enhanced_results_7pt(self, img, pts, state, start_idx):
        img_out = img.copy()
        if len(pts) < 7: return img_out
        quads = self.get_quads(pts)
        
        for i, quad in enumerate(quads):
            q = np.array(quad, np.int32)
            cv2.polylines(img_out, [q], True, (0, 255, 255), 2)
            
            src_pts = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
            dst_pts = np.float32(quad)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            for r in range(3):
                for c in range(3):
                    p = np.array([[[float(c) + 0.5, float(r) + 0.5]]], dtype=np.float32)
                    p_trans = cv2.perspectiveTransform(p, M)[0][0]
                    
                    if np.any(np.isnan(p_trans)): continue
                    
                    px, py = int(p_trans[0]), int(p_trans[1])
                    
                    color_idx = start_idx + i*9 + r*3 + c
                    color_name = state[color_idx]
                    rgb = self.get_color_rgb(color_name)
                    
                    # Gradio image is RGB. Draw in RGB.
                    cv2.circle(img_out, (px, py), 13, rgb, -1)
                    cv2.circle(img_out, (px, py), 13, (50, 50, 50), 2) # Dark border
                    
                    # Contrast for text: use Black for white/yellow, White for others
                    txt_col = (0, 0, 0)
                    if color_name in ['blue', 'red', 'green']:
                        txt_col = (255, 255, 255)
                        
                    cv2.putText(img_out, color_name[0].upper(), (px-7, py+8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, txt_col, 2)
        return img_out

    def get_color_rgb(self, name):
        hex_val = COLOR_VALUES.get(name, "#888888").lstrip('#')
        return tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4)) # Standard RGB

    def solve(self, current_state):
        try:
            faces_colors = {}
            for i, face in enumerate(FACE_NAMES):
                faces_colors[face] = current_state[i*9 : (i+1)*9]
                # Log state of each face
                self.log(f"面 {face} 状态: {','.join(faces_colors[face])}")
            
            cube_str = map_colors_to_string(faces_colors)
            solution = solve_cube(cube_str)
            return self.log(f"还原方案: {solution}"), solution
        except Exception as e:
            return self.log(f"解算异常: {str(e)}"), "无法解算"

    def reset_state(self):
        new_state = INITIAL_STATE.copy()
        self.clicks_a = []
        self.clicks_b = []
        updates = []
        for color in new_state:
            updates.append(gr.update(elem_classes=["sticker-btn", f"color-{color}"]))
        return [new_state, self.log("状态与点击坐标已全量重置"), "", None, None] + updates + [self.get_3d_update_js(new_state)]

app_logic = RubiksApp()

CSS = """
.sticker-btn { border: 2px solid #222 !important; min-width: 40px !important; height: 40px !important; margin: 1px !important; padding: 0 !important; }
.face-container { padding: 5px; border: 1px solid #555; background: #333; }
.color-white { background-color: #FFFFFF !important; color: black !important; }
.color-yellow { background-color: #FFFF00 !important; color: black !important; }
.color-red { background-color: #FF0000 !important; }
.color-orange { background-color: #FFA500 !important; }
.color-green { background-color: #008000 !important; }
.color-blue { background-color: #0000FF !important; }
"""

JS_CODE = """
function() {
    window.updateCube3D = function(state) {
        const iframe = document.getElementById('cube-3d-iframe');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({ type: 'UPDATE_STATE', state: state }, '*');
        }
    };
    // Initialize 3D view after a short delay to ensure iframe is loaded
    setTimeout(() => {
        if (window.updateCube3D) {
            // Get initial state if possible or wait for first update
        }
    }, 1000);
}
"""

with gr.Blocks() as demo:
    state = gr.State(INITIAL_STATE)
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("第一步：图像捕捉"):
                    gr.Markdown("### 📸 7点定位引导\n1. 请在视角 A 中逆时针点击 6 个外围顶点，最后点击中心点 7。\n2. 视角 B 同样按此顺序标注。")
                    with gr.Row():
                        with gr.Column():
                            img_a = gr.Image(label="视角 A (顶-前-右)", value="U-F-R.png" if os.path.exists("U-F-R.png") else None)
                        with gr.Column():
                            img_b = gr.Image(label="视角 B (底-后-左)", value="D-B-L.png" if os.path.exists("D-B-L.png") else None)
                    
                    with gr.Row():
                        btn_identify = gr.Button("🔍 提取色彩识别", variant="primary")
                        btn_clear_pts = gr.Button("🗑️ 清空点击点")
                
                with gr.TabItem("第二步：校验与手动修改"):
                    gr.Markdown("### 🧱 展开图纠错 (点击色块切换颜色)")
                    
                    # Layout: 
                    #       [U]
                    #   [L] [F] [R] [B]
                    #       [D]
                    
                    sticker_btns = []
                    
                    with gr.Column():
                        # U Face
                        with gr.Row():
                            gr.Markdown("")
                            with gr.Column(scale=1):
                                gr.Markdown("<center><b>U (顶面)</b></center>")
                                with gr.Group(elem_classes="face-container"):
                                    for r in range(3):
                                        with gr.Row():
                                            for c in range(3):
                                                idx = 0 + r*3 + c
                                                btn = gr.Button("", elem_classes=["sticker-btn", f"color-{INITIAL_STATE[idx]}"])
                                                sticker_btns.append(btn)
                            gr.Markdown("")
                        
                        # L, F, R, B Faces
                        with gr.Row():
                            for face_idx, face_name in enumerate(['L', 'F', 'R', 'B']):
                                face_base = [36, 18, 9, 45][face_idx] # Indices in 54-char string
                                with gr.Column(scale=1):
                                    gr.Markdown(f"<center><b>{face_name}</b></center>")
                                    with gr.Group(elem_classes="face-container"):
                                        for r in range(3):
                                            with gr.Row():
                                                for c in range(3):
                                                    idx = face_base + r*3 + c
                                                    btn = gr.Button("", elem_classes=["sticker-btn", f"color-{INITIAL_STATE[idx]}"])
                                                    sticker_btns.append(btn)
                        
                        # D Face
                        with gr.Row():
                            gr.Markdown("")
                            with gr.Column(scale=1):
                                gr.Markdown("<center><b>D (底面)</b></center>")
                                with gr.Group(elem_classes="face-container"):
                                    for r in range(3):
                                        with gr.Row():
                                            for c in range(3):
                                                idx = 27 + r*3 + c
                                                btn = gr.Button("", elem_classes=["sticker-btn", f"color-{INITIAL_STATE[idx]}"])
                                                sticker_btns.append(btn)
                            gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### 🧊 3D 实况预览")
            gr.HTML(
                '<div style="height: 350px; background: #111; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">' +
                f'<iframe id="cube-3d-iframe" srcdoc="{SAFE_CUBE_HTML}" style="width: 100%; height: 100%; border: none;"></iframe>' +
                '</div>'
            )
            
            with gr.Column():
                gr.Markdown("### ⚙️ 控制面板")
                btn_solve = gr.Button("🚀 开始智能求解", variant="primary")
                btn_reset = gr.Button("🔄 重置状态")
                out_steps = gr.Textbox(label="还原公式", lines=2, interactive=False)
                
            log_display = gr.TextArea(value="[系统就绪] 请在上方视角 A 中点击第一个点：最上方顶点 (顶面后角)", interactive=False, lines=8)

    # Handlers
    img_a.select(app_logic.handle_click_a, inputs=[img_a], outputs=[img_a, log_display])
    img_b.select(app_logic.handle_click_b, inputs=[img_b], outputs=[img_b, log_display])
    btn_clear_pts.click(app_logic.clear_clicks, outputs=[img_a, img_b, log_display])
    
    # To call JS from python in Gradio 4+, we use _js in a component or gr.Interface.
    # Better: use a dummy hidden textbox that triggers JS on change.
    js_trigger = gr.Textbox(visible=False)
    js_trigger.change(None, inputs=[js_trigger], outputs=None, js="(val) => { if(val) eval(val); }")

    # Canonical order (U, R, F, D, L, B):
    # U: 0-8, R: 9-17, F: 18-26, D: 27-35, L: 36-44, B: 45-53
    ordered_btns = sticker_btns[0:9] # U
    ordered_btns += sticker_btns[27:36] # R
    ordered_btns += sticker_btns[18:27] # F
    ordered_btns += sticker_btns[45:54] # D
    ordered_btns += sticker_btns[9:18] # L
    ordered_btns += sticker_btns[36:45] # B

    btn_identify.click(
        app_logic.identify_colors, 
        inputs=[img_a, img_b, state], 
        outputs=[state, log_display, js_trigger, img_a, img_b] + ordered_btns
    )
    ordered_btns = sticker_btns[0:9] # U
    ordered_btns += sticker_btns[27:36] # R
    ordered_btns += sticker_btns[18:27] # F
    ordered_btns += sticker_btns[45:54] # D
    ordered_btns += sticker_btns[9:18] # L
    ordered_btns += sticker_btns[36:45] # B

    for i in range(54):
        def make_handler(idx):
            return lambda s: app_logic.update_sticker(idx, s)
        
        ordered_btns[i].click(
            fn=make_handler(i),
            inputs=[state],
            outputs=[state, ordered_btns[i], js_trigger]
        )

    btn_solve.click(app_logic.solve, inputs=[state], outputs=[log_display, out_steps])
    
    # Reset is tricky because it needs to update 54 buttons.
    btn_reset.click(
        fn=app_logic.reset_state,
        outputs=[state, log_display, out_steps, img_a, img_b] + ordered_btns + [js_trigger]
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["."], css=CSS, js=JS_CODE)
