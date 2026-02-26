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
FACE_NAMES = ['U', 'F', 'R', 'D', 'L', 'B']

# Initial solved state (54 colors)
INITIAL_STATE = []
for face in ['white', 'green', 'red', 'yellow', 'blue', 'orange']:
    INITIAL_STATE.extend([face] * 9)

# Kociemba standard string order: U1..U9, R1..R9, F1..F9, D1..D9, L1..L9, B1..B9

def rotate_face_cw(face_arr):
    # Rotate 3x3 array clockwise
    return [
        face_arr[6], face_arr[3], face_arr[0],
        face_arr[7], face_arr[4], face_arr[1],
        face_arr[8], face_arr[5], face_arr[2]
    ]

def rotate_cube_state(state, move):
    state = list(state)
    face = move[0]
    mod = move[1:] if len(move) > 1 else ""
    
    # 0=U, 9=F, 18=R, 27=D, 36=L, 45=B
    face_offsets = {'U':0, 'F':9, 'R':18, 'D':27, 'L':36, 'B':45}
    base = face_offsets[face]
    
    # Adjacency cycles (CW edge mapping)
    cycles = {
        'U': ([47,46,45], [20,19,18], [11,10,9], [38,37,36]),
        'D': ([15,16,17], [24,25,26], [51,52,53], [42,43,44]),
        'F': ([6,7,8], [18,21,24], [29,28,27], [44,41,38]),
        'B': ([2,1,0], [36,39,42], [33,34,35], [26,23,20]),
        'R': ([8,5,2], [45,48,51], [35,32,29], [17,14,11]),
        'L': ([0,3,6], [9,12,15], [27,30,33], [53,50,47])
    }
    
    times = 1
    if mod == "'": times = 3
    elif mod == "2": times = 2
    
    for _ in range(times):
        # Rotate face
        state[base:base+9] = rotate_face_cw(state[base:base+9])
        
        # Rotate adjacent edges
        c = cycles[face]
        tmp = [state[i] for i in c[3]]
        for i in range(3): state[c[3][i]] = state[c[2][i]]
        for i in range(3): state[c[2][i]] = state[c[1][i]]
        for i in range(3): state[c[1][i]] = state[c[0][i]]
        for i in range(3): state[c[0][i]] = tmp[i]
        
    return state


class RubiksApp:
    CLICKS_FILE = 'saved_clicks.json'

    def __init__(self):
        self.logs = []
        self.clicks_a = []
        self.clicks_b = []
        self.orig_img_a = None
        self.orig_img_b = None
        self._try_load_clicks()

    def _try_load_clicks(self):
        """Auto-load saved clicks if file exists."""
        try:
            if os.path.exists(self.CLICKS_FILE):
                with open(self.CLICKS_FILE, 'r') as f:
                    data = json.load(f)
                self.clicks_a = [tuple(p) for p in data.get('clicks_a', [])]
                self.clicks_b = [tuple(p) for p in data.get('clicks_b', [])]
                print(f"Loaded saved clicks: A={len(self.clicks_a)}, B={len(self.clicks_b)}")
        except Exception as e:
            print(f"Could not load saved clicks: {e}")

    def save_clicks(self):
        """Save current clicks to JSON file."""
        data = {'clicks_a': self.clicks_a, 'clicks_b': self.clicks_b}
        with open(self.CLICKS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return self.log(f"已保存标注点: A={len(self.clicks_a)}点, B={len(self.clicks_b)}点")

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
        import time
        state_dict = {}
        for i, face in enumerate(FACE_NAMES):
            state_dict[face] = state[i*9 : (i+1)*9]
        return (
            f"(function(){{ "
            f"console.log('Execute 3D UI update, ts: {time.time()}');"
            f"var stateObj = {json.dumps(state_dict)};"
            f"var iframes = document.querySelectorAll('iframe');"
            f"for(var i=0; i<iframes.length; i++){{"
            f"  try {{ iframes[i].contentWindow.postMessage({{ type: 'UPDATE_STATE', state: stateObj }}, '*'); }} catch(e){{}}"
            f"}}"
            f"}})()"
        )

    def get_3d_animation_js(self, state, move_str):
        state_dict = {}
        for i, face in enumerate(FACE_NAMES):
            state_dict[face] = state[i*9 : (i+1)*9]
        state_json = json.dumps(state_dict)
        return (
            f"(function(){{ "
            f"var iframes = document.querySelectorAll('iframe'); "
            f"for(var i=0; i<iframes.length; i++) {{ "
            f"  try {{ iframes[i].contentWindow.postMessage({{ type: 'MOVE_AND_UPDATE', move: '{move_str}', state: {state_json} }}, '*'); }} catch(e){{}} "
            f"}} "
            f"}})()"
        )

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
            return img, self.log("提示: 视角 A 已标注 7 个点。如需重选请点击 [清空点击点]")
        self.clicks_a.append((evt.index[0], evt.index[1]))
        return self.draw_and_guide(img, self.clicks_a, "A")

    def handle_click_b(self, img, evt: gr.SelectData):
        if len(self.clicks_b) >= 7:
            return img, self.log("提示: 视角 B 已标注 7 个点。如需重选请点击 [清空点击点]")
        self.clicks_b.append((evt.index[0], evt.index[1]))
        return self.draw_and_guide(img, self.clicks_b, "B")

    def set_orig_a(self, img):
        if img is not None:
            self.orig_img_a = img.copy()
            self.clicks_a = []
            return self.log("视角 A 图片已导入，标注点已自?重置")
        return gr.update()

    def set_orig_b(self, img):
        if img is not None:
            self.orig_img_b = img.copy()
            self.clicks_b = []
            return self.log("视角 B 图片已导入，标注点已自?重置")
        return gr.update()

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
        # Return clean images if we have them
        res_a = self.orig_img_a if self.orig_img_a is not None else gr.update()
        res_b = self.orig_img_b if self.orig_img_b is not None else gr.update()
        return res_a, res_b, self.log("📌 点点击坐标已清空，视图已恢复原始状态")

    def identify_colors(self, img_a, img_b, current_state):
        if img_a is None or img_b is None:
            return [current_state, self.log("错误: 请先上传或捕捉两张魔方视角图片"), "", gr.update(), gr.update()] + [gr.update()]*54
        
        print(f"DEBUG: identify_colors called. Image A: {getattr(img_a, 'shape', 'no shape')}, Image B: {getattr(img_b, 'shape', 'no shape')}")
        print(f"DEBUG: Clicks A: {len(self.clicks_a)}, Clicks B: {len(self.clicks_b)}")

        if len(self.clicks_a) < 7 or len(self.clicks_b) < 7:
            return [current_state, self.log(f"提示: 需要标注 7 个定位点 (当前 A:{len(self.clicks_a)} B:{len(self.clicks_b)})"), "", gr.update(), gr.update()] + [gr.update()]*54

        try:
            # Use original clean images for sampling to avoid UI markers
            s_img_a = self.orig_img_a if self.orig_img_a is not None else img_a
            s_img_b = self.orig_img_b if self.orig_img_b is not None else img_b

            self.log("步骤 1/4: 采样视角 A 颜色...")
            raw_colors_a = self.sample_colors_7pt(s_img_a, self.clicks_a, 'A', 0)
            
            self.log("步骤 2/4: 采样视角 B 颜色...")
            raw_colors_b = self.sample_colors_7pt(s_img_b, self.clicks_b, 'B', 27)
            
            all_raw = raw_colors_a + raw_colors_b
            
            self.log("步骤 3/4: 分析色彩分布与聚类匹配...")
            # cv_engine returns colors in order: [U(0-8), F(9-17), R(18-26), D(27-35), L(36-44), B(45-53)]
            engine_state, _, _ = cluster_and_map_colors(all_raw)
            
            new_state = list(engine_state)
            
            self.log("步骤 4/4: 生成 3D 预览与校验视图...")
            res_a = self.draw_enhanced_results_7pt(img_a, self.clicks_a, engine_state, 'A', 0)
            res_b = self.draw_enhanced_results_7pt(img_b, self.clicks_b, engine_state, 'B', 27)
            
            # Generate UI updates for the 54 buttons (mapped via ordered_btns)
            btn_updates = []
            for color in new_state:
                # Add a label as value to confirm the color update is happening
                label = color[0].upper()
                btn_updates.append(gr.update(value=label, elem_classes=["sticker-btn", f"color-{color}"]))

            final_log = self.log("✅ 识别成功! 魔方状态已更新到 3D 预览。")
            
            # Update 3D
            js_update = self.get_3d_update_js(new_state)
            
            return [new_state, final_log, js_update, res_a, res_b] + btn_updates
        except Exception as e:
            import traceback
            traceback.print_exc()
            return [current_state, self.log(f"❌ 识别发生内部错误: {str(e)}"), "", gr.update(), gr.update()] + [gr.update()]*54

    def get_quads(self, pts, view_type='A'):
        # pts[0]=1(top), pts[1]=2(NW), pts[2]=3(SW), pts[3]=4(bottom), pts[4]=5(SE), pts[5]=6(NE), pts[6]=7(center)
        if view_type == 'A':
            # View A returns: [Top face, Left-side face, Right-side face]
            #   = [U, F, R] in physical cube terms
            return [
                (pts[0], pts[5], pts[6], pts[1]), # U
                (pts[1], pts[6], pts[3], pts[2]), # F
                (pts[6], pts[5], pts[4], pts[3])  # R
            ]
        else:
            # View B returns: [Top face, Left-side face, Right-side face]  
            #   = [D, L, B] in physical cube terms
            # Note: Because View B is upside down relative to U, the corners mapping requires
            # flipping the quad array to keep (row, col) compliant with Kociemba standard.
            return [
                (pts[1], pts[0], pts[5], pts[6]), # D: Top-Left=pt2, Top-Right=pt1...
                (pts[3], pts[2], pts[1], pts[6]), # L: Top-Left=pt4, Top-Right=pt3...
                (pts[4], pts[3], pts[6], pts[5])  # B: Top-Left=pt5, Top-Right=pt4...
            ]


    def sample_colors_7pt(self, img, pts, view_type, start_idx):
        quads = self.get_quads(pts, view_type)
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

    def draw_enhanced_results_7pt(self, img, pts, state, view_type, start_idx):
        img_out = img.copy()
        if len(pts) < 7: return img_out
        quads = self.get_quads(pts, view_type)
        
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
            
            # Print center colors for debug
            centers = {f: faces_colors[f][4] for f in FACE_NAMES}
            self.log(f"中心颜色: {centers}")
            
            cube_str = map_colors_to_string(faces_colors)
            self.log(f"Kociemba字符串: {cube_str}")
            
            solution = solve_cube(cube_str)
            return self.log(f"还原方案: {solution}"), solution
        except Exception as e:
            return self.log(f"解算异常: {str(e)}"), "无法解算"

    def scramble(self, state):
        import random
        import time
        moves_list = ['U', 'D', 'F', 'B', 'L', 'R']
        mods = ['', "'", '2']
        sequence = [random.choice(moves_list) + random.choice(mods) for _ in range(20)]
        self.log(f"正在打乱: {' '.join(sequence)}")
        
        for move in sequence:
            state = rotate_cube_state(state, move)
            updates = [gr.update(elem_classes=["sticker-btn", f"color-{c}"]) for c in state]
            js = self.get_3d_animation_js(state, move)
            yield [state, self.log(f"打乱: {move}")] + updates + [js]
            time.sleep(0.35) 
            
    def play_solution(self, state, steps_str):
        import time
        if not steps_str or steps_str == "无法解算":
            yield [state, self.log("提示: 请先生成有效的解算公式！")] + [gr.update()]*54 + [None]
            return
            
        steps = steps_str.split()
        for move in steps:
            state = rotate_cube_state(state, move)
            updates = [gr.update(elem_classes=["sticker-btn", f"color-{c}"]) for c in state]
            js = self.get_3d_animation_js(state, move)
            yield [state, self.log(f"播放执行: {move}")] + updates + [js]
            time.sleep(0.35)
        yield [state, self.log("🎉 还原动画播放完毕！")] + updates + [None]

    def handle_keyboard_move(self, state, move_str):
        if not move_str: return [state, gr.update(), gr.update()] + [gr.update()]*54 + [None]
        
        actual_move = move_str.split('_')[0]
        new_state = rotate_cube_state(state, actual_move)
        logs = self.log(f"⌨️ 键盘旋转: {actual_move}")
        updates = [gr.update(elem_classes=["sticker-btn", f"color-{c}"]) for c in new_state]
        
        # We do not clear move_str anymore to prevent double-firing. The JS makes it unique.
        return [new_state, logs, gr.update()] + updates + [self.get_3d_update_js(new_state)]

    def save_state(self, state):
        self.saved_state = list(state)
        return self.log("💾 当前状态已保存，可随时使用 [恢复状态] 还原。")

    def restore_state(self):
        if hasattr(self, 'saved_state') and self.saved_state:
            updates = [gr.update(elem_classes=["sticker-btn", f"color-{c}"]) for c in self.saved_state]
            js = self.get_3d_update_js(self.saved_state)
            return [self.saved_state, self.log("📂 状态已恢复到保存点！")] + updates + [js]
        else:
            return [gr.update(), self.log("提示: 系统中暂无保存的状态。")] + [gr.update()]*54 + [None]

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
.hidden-bridge {
    position: absolute !important;
    left: -9999px !important;
    height: 0 !important;
    overflow: hidden !important;
}
.color-white { background-color: #FFFFFF !important; color: black !important; }
.color-yellow { background-color: #FFFF00 !important; color: black !important; }
.color-red { background-color: #FF0000 !important; }
.color-orange { background-color: #FFA500 !important; }
.color-green { background-color: #008000 !important; }
.color-blue { background-color: #0000FF !important; }
"""

JS_CODE = """
function() {
    console.log("Rubiks Bridge: Initializing...");
    
    window.addEventListener('message', function(event) {
        if (event.data.type === 'SEND_MOVE') {
            const move = event.data.move;
            console.log("Rubiks Bridge: Move request ->", move);
            
            const inputWrapper = document.getElementById('hidden-move-input');
            if (inputWrapper) {
                const textarea = inputWrapper.querySelector('textarea');
                if (textarea) {
                    textarea.value = move + "_" + Date.now();
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    textarea.dispatchEvent(new Event('change', { bubbles: true }));
                    console.log("Rubiks Bridge: Dispatched change event");
                }
            }
        }
    });

    window.updateCube3D = function(state) {
        const iframe = document.getElementById('cube-3d-iframe');
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage({ type: 'UPDATE_STATE', state: state }, '*');
        }
    };
}
"""

# Pre-draw saved clicks on images for initial display
init_img_a = "U-F-R.png" if os.path.exists("U-F-R.png") else None
init_img_b = "D-B-L.png" if os.path.exists("D-B-L.png") else None

if app_logic.clicks_a and init_img_a:
    _img = cv2.imread(init_img_a)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    app_logic.orig_img_a = _img.copy() # Keep clean copy
    init_img_a, _ = app_logic.draw_and_guide(_img, app_logic.clicks_a, "A")
if app_logic.clicks_b and init_img_b:
    _img = cv2.imread(init_img_b)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    app_logic.orig_img_b = _img.copy() # Keep clean copy
    init_img_b, _ = app_logic.draw_and_guide(_img, app_logic.clicks_b, "B")

with gr.Blocks() as demo:
    state = gr.State(INITIAL_STATE)
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("第一步：图像捕捉"):
                    gr.Markdown("### 📸 7点定位引导\n1. 请在视角 A 中逆时针点击 6 个外围顶点，最后点击中心点 7。\n2. 视角 B 同样按此顺序标注。")
                    with gr.Row():
                        with gr.Column():
                            img_a = gr.Image(label="视角 A (顶-前-右)", value=init_img_a)
                        with gr.Column():
                            img_b = gr.Image(label="视角 B (底-后-左)", value=init_img_b)
                    
                    with gr.Row():
                        btn_identify = gr.Button("🔍 提取色彩识别", variant="primary")
                        btn_save_pts = gr.Button("💾 保存标注点")
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
                                face_base = [36, 9, 18, 45][face_idx] # L=36, F=9, R=18, B=45 in state
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
                
                with gr.Row():
                    btn_solve = gr.Button("🚀 计算还原公式", variant="primary")
                    btn_play = gr.Button("▶️ 播放还原动画")
                    
                out_steps = gr.Textbox(label="还原公式 (空格分隔序列)", lines=2, interactive=False)
                
                with gr.Row():
                    btn_save_state = gr.Button("💾 保存状态")
                    btn_restore_state = gr.Button("📂 恢复状态")
                    
                with gr.Row():
                    btn_scramble = gr.Button("🔀 一键随机打乱")
                    btn_reset = gr.Button("🔄 重置为白顶绿前")
                
            # Hidden elements for JS keyboard bridge
            hidden_move_input = gr.Textbox(label="Bridge Input", elem_id="hidden-move-input", elem_classes="hidden-bridge")
                
            log_display = gr.TextArea(
                value=f"[系统就绪] {'已加载保存的标注点 (A={len(app_logic.clicks_a)}, B={len(app_logic.clicks_b)}), 可直接提取!' if app_logic.clicks_a else '请在上方视角 A 中点击第一个点'}",
                interactive=False, lines=8)

    # Handlers
    img_a.upload(app_logic.set_orig_a, inputs=[img_a], outputs=[log_display])
    img_b.upload(app_logic.set_orig_b, inputs=[img_b], outputs=[log_display])
    
    img_a.select(app_logic.handle_click_a, inputs=[img_a], outputs=[img_a, log_display])
    img_b.select(app_logic.handle_click_b, inputs=[img_b], outputs=[img_b, log_display])
    btn_clear_pts.click(app_logic.clear_clicks, outputs=[img_a, img_b, log_display])
    btn_save_pts.click(app_logic.save_clicks, outputs=[log_display])
    
    # To call JS from python in Gradio 4+, we use _js in a component or gr.Interface.
    # Better: use a dummy hidden textbox that triggers JS on change.
    js_trigger = gr.Textbox(visible=False)
    js_trigger.change(None, inputs=[js_trigger], outputs=None, js="(val) => { if(val) eval(val); }")

    # State order: U(0-8), F(9-17), R(18-26), D(27-35), L(36-44), B(45-53)
    # sticker_btns creation order: [0:9]=U, [9:18]=L(36), [18:27]=F(9), [27:36]=R(18), [36:45]=B(45), [45:54]=D(27)
    # ordered_btns maps sticker_btns to state order
    ordered_btns = sticker_btns[0:9]   # U → state 0-8
    ordered_btns += sticker_btns[18:27] # F → state 9-17
    ordered_btns += sticker_btns[27:36] # R → state 18-26
    ordered_btns += sticker_btns[45:54]  # D → state 27-35
    ordered_btns += sticker_btns[9:18]   # L → state 36-44
    ordered_btns += sticker_btns[36:45]  # B → state 45-53

    # Bind Keyboard Events via invisible proxy input change
    hidden_move_input.change(
        fn=app_logic.handle_keyboard_move,
        inputs=[state, hidden_move_input],
        outputs=[state, log_display, hidden_move_input] + ordered_btns + [js_trigger]
    )

    btn_identify.click(
        app_logic.identify_colors, 
        inputs=[img_a, img_b, state], 
        outputs=[state, log_display, js_trigger, img_a, img_b] + ordered_btns
    )

    for i in range(54):
        def make_handler(idx):
            return lambda s: app_logic.update_sticker(idx, s)
        
        ordered_btns[i].click(
            fn=make_handler(i),
            inputs=[state],
            outputs=[state, ordered_btns[i], js_trigger]
        )

    btn_solve.click(app_logic.solve, inputs=[state], outputs=[log_display, out_steps])
    
    # Interactions
    btn_play.click(
        fn=app_logic.play_solution,
        inputs=[state, out_steps],
        outputs=[state, log_display] + ordered_btns + [js_trigger]
    )
    
    btn_scramble.click(
        fn=app_logic.scramble,
        inputs=[state],
        outputs=[state, log_display] + ordered_btns + [js_trigger]
    )
    
    btn_save_state.click(app_logic.save_state, inputs=[state], outputs=[log_display])
    
    btn_restore_state.click(
        fn=app_logic.restore_state,
        outputs=[state, log_display] + ordered_btns + [js_trigger]
    )
    
    # Reset is tricky because it needs to update 54 buttons.
    btn_reset.click(
        fn=app_logic.reset_state,
        outputs=[state, log_display, out_steps, img_a, img_b] + ordered_btns + [js_trigger]
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["."], css=CSS, js=JS_CODE)
