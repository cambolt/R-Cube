import cv2
import numpy as np

# A standard rubik's cube face has 3x3 smaller squares. 
# We'll warp each visible face to a 300x300 square to sample colors.
WARPED_SIZE = 300
CELL_SIZE = WARPED_SIZE // 3

# Hue ranges for standard Rubik's cube colors (Approximate, requires tuning based on lighting)
# We will use K-Means clustering instead for better robustness, but keep this for reference or fallback.
COLOR_RANGES = {
    'red':    [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
    'orange': [(11, 70, 50), (25, 255, 255)],
    'yellow': [(26, 70, 50), (35, 255, 255)],
    'green':  [(36, 70, 50), (85, 255, 255)],
    'blue':   [(86, 70, 50), (130, 255, 255)],
    'white':  [(0, 0, 150), (180, 60, 255)] # Low saturation, high value
}

def get_perspective_transform(points, face_type):
    """
    Given 3 points clicked by the user on a face (e.g. Center, Top-Left, Bottom-Right),
    or bounding points, calculate the perspective transform matrix.
    To make it simple for the user, we ask for:
    1. Top point (or top-left corner)
    2. Bottom-left point
    3. Bottom-right point
    Wait, the user requirement says: "辅助定位: 用户点击3个中心色块(或魔方外轮廓关键角点)".
    Let's implement a 4-point perspective warp. We need 4 points per face.
    For simplicity in a 2D view of a 3D isometric cube (showing 3 faces U, F, R):
    
    Center of the cube (where U, F, R meet) is a common point.
    To extract Face U: points needed are (Center, TopLeft_U, TopRight_U, BottomLeft_U = which is Center for U).
    Actually, clicking 3 center pieces of U, F, R gives us the orientation.
    But to extract the grid, we need the exact corners of each face.
    
    Let's refine: The user clicks the 3 outer corners and 1 central inner corner (where 3 faces meet) = 4 points per face.
    Even better: User clicks 7 points total on the image to bound the 3 faces:
    1: Central intersection (where U, F, R meet)
    2, 3, 4: The 3 outer corners of the faces
    5, 6, 7: The furthest extreme corners
    
    For MVP, we'll ask the user to click 4 corners for a specific face to extract it.
    Or, if user clicks the 3 centers, we can estimate the bounding box. But estimating from 3 centers is mathematically ambiguous without knowing the exact projection.
    
    Let's stick to conventional 4-point selection per face, or 6 points for the outer hexagon + 1 center point.
    If the user clicks:
    p0: center (meets U,F,R)
    p1: top (above U)
    p2: right (right of R)
    p3: bottom (below F)
    p4: top-left (left of U)
    p5: bottom-left (left of F)
    p6: top-right (above R)
    
    Let's implement a universal function that takes 4 corners of a face (TopLeft, TopRight, BottomRight, BottomLeft) and warps it.
    """
    pts1 = np.float32(points)
    # Warping to a square
    pts2 = np.float32([
        [0, 0], 
        [WARPED_SIZE, 0], 
        [WARPED_SIZE, WARPED_SIZE], 
        [0, WARPED_SIZE]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix

def extract_face(image, points):
    """
    Extracts a perfectly square 300x300 face from the image using 4 corner points.
    points: List of 4 (x,y) tuples: [TopLeft, TopRight, BottomRight, BottomLeft]
    """
    matrix = get_perspective_transform(points, "any")
    warped = cv2.warpPerspective(image, matrix, (WARPED_SIZE, WARPED_SIZE))
    return warped

def sample_colors_from_face(warped_face):
    """
    Given a 300x300 warped face, samples the 9 cells.
    Returns a list of 9 BGR color tuples (or RGB depending on input image).
    """
    colors = []
    offset = CELL_SIZE // 2 # Center of the cell
    
    for row in range(3):
        for col in range(3):
            cy = row * CELL_SIZE + offset
            cx = col * CELL_SIZE + offset
            
            # Sample a small region around the center to average out noise
            region = warped_face[cy-5:cy+5, cx-5:cx+5]
            avg_color = np.median(region, axis=(0, 1))
            colors.append(tuple(map(int, avg_color)))
            
    return colors

def cluster_and_map_colors(all_sampled_colors):
    """
    Advanced recognition using HSV space, face normalization, and center-referencing.
    """
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import itertools

    # 1. Convert RGB to HSV for better lighting handling
    # IMPORTANT: Gradio provides images in RGB format, NOT BGR!
    data_rgb = np.uint8([all_sampled_colors])
    data_hsv = cv2.cvtColor(data_rgb, cv2.COLOR_RGB2HSV)[0].astype(np.float32)
    data_rgb_f = data_rgb[0].astype(np.float32)

    # Center indices in ACTUAL sampling order:
    # View A quad order: U(0-8), F(9-17), R(18-26)
    # View B quad order: D(27-35), B(36-44), L(45-53)
    # Center = index 4 within each group of 9
    # So: U_center=4, F_center=13, R_center=22, D_center=31, B_center=40, L_center=49
    CENTER_IDXS = [4, 13, 22, 31, 40, 49]  # U, F, R, D, B, L
    # face_id mapping: 0=U, 1=F, 2=R, 3=D, 4=B, 5=L
    center_hsv = data_hsv[CENTER_IDXS]
    center_rgb = data_rgb_f[CENTER_IDXS]

    # Debug: print center RGB and HSV values
    face_labels = ['U', 'F', 'R', 'D', 'B', 'L']
    for k in range(6):
        print(f"  Center {face_labels[k]}: RGB=({int(center_rgb[k][0])},{int(center_rgb[k][1])},{int(center_rgb[k][2])}) "
              f"HSV=({center_hsv[k][0]:.1f},{center_hsv[k][1]:.1f},{center_hsv[k][2]:.1f})")


    # Map centers to standard names.
    # Strategy:
    #   Step 1: White = lowest saturation (hue is meaningless for near-gray colors)
    #   Step 2: Remaining 5 → assign by hue using Hungarian matching
    #           Red is special: it lives near hue 0 AND hue 180 on the HSV wheel.
    face_names = [""] * 6
    
    # Step 1: White has the lowest saturation
    sat_vals = [(float(center_hsv[i][1]), i) for i in range(6)]
    white_center_idx = min(sat_vals, key=lambda x: x[0])[1]
    face_names[white_center_idx] = "white"
    remaining_centers = [i for i in range(6) if i != white_center_idx]
    
    # Step 2: Match the other 5 by hue
    # OpenCV hue: yellow≈25-30, orange≈10-15, red≈0 or 160-180, green≈55-85, blue≈100-120
    hue_targets = {'yellow': 28.0, 'orange': 12.0, 'red': 0.0, 'green': 65.0, 'blue': 110.0}
    color_order = list(hue_targets.keys())
    n = len(remaining_centers)  # = 5
    cost_hue = np.zeros((n, n))
    for i, c_idx in enumerate(remaining_centers):
        h = float(center_hsv[c_idx][0])
        for j, name in enumerate(color_order):
            target_h = hue_targets[name]
            if name == 'red':
                # Red wraps: min distance to 0 considering 180-wraparound
                h_dist = min(h, 180.0 - h)
            else:
                raw = abs(h - target_h)
                h_dist = min(raw, 180.0 - raw)
            cost_hue[i, j] = h_dist
    
    from scipy.optimize import linear_sum_assignment
    r_ind, c_ind = linear_sum_assignment(cost_hue)
    for r, c in zip(r_ind, c_ind):
        face_names[remaining_centers[r]] = color_order[c]

    # Debug: show face name mapping
    fnl = ['U', 'F', 'R', 'D', 'B', 'L']
    for i in range(6):
        print(f"  face_names[{i}]({fnl[i]}) = {face_names[i]}")

    # 2. Color matching using combined HSV + RGB features
    # HSV hue for general color discrimination (blue vs green vs red etc.)
    # RGB G/R ratio for yellow vs orange (stable across lighting):
    #   Yellow: G/R ≈ 0.85-0.95 (G is close to R)
    #   Orange: G/R ≈ 0.60-0.80 (G is much lower than R)
    
    # Compute G/R ratio for each center
    center_gr_ratio = []
    for k in range(6):
        r_val = max(float(center_rgb[k][0]), 1.0)
        g_val = float(center_rgb[k][1])
        center_gr_ratio.append(g_val / r_val)
        print(f"  Center {['U','F','R','D','B','L'][k]} ({face_names[k]}): G/R={center_gr_ratio[-1]:.3f}")
    
    def color_dist(hsv_sticker, rgb_sticker, center_id):
        """Distance using HSV hue + RGB G/R ratio for yellow/orange."""
        h1, s1, v1 = float(hsv_sticker[0]), float(hsv_sticker[1]), float(hsv_sticker[2])
        h2, s2, v2 = float(center_hsv[center_id][0]), float(center_hsv[center_id][1]), float(center_hsv[center_id][2])
        
        # White detection: saturation-based
        if face_names[center_id] == 'white':
            return abs(s1 - s2) * 2.0 + abs(v1 - v2) * 0.3
        
        # Low-saturation sticker trying to match chromatic color = penalty
        if s1 < 40:
            return 500.0
        
        # Cyclic hue distance
        h_dist = min(abs(h1 - h2), 180.0 - abs(h1 - h2))
        
        # Yellow / Orange special handling using G/R ratio
        if face_names[center_id] in ('yellow', 'orange') and h1 < 35:
            r_val = max(float(rgb_sticker[0]), 1.0)
            g_val = float(rgb_sticker[1])
            sticker_gr = g_val / r_val
            center_gr = center_gr_ratio[center_id]
            
            # G/R ratio distance is the primary discriminator for yellow vs orange
            gr_dist = abs(sticker_gr - center_gr)
            return h_dist * 1.0 + gr_dist * 300.0 + abs(v1 - v2) * 0.1
        
        return h_dist * 3.0 + abs(s1 - s2) * 0.5 + abs(v1 - v2) * 0.2

    res = [""] * 54
    for i in range(6): res[CENTER_IDXS[i]] = face_names[i]

    # Debug: dump all 54 stickers
    face_label_map = ['U', 'F', 'R', 'D', 'B', 'L']
    print("  === All 54 sticker samples ===")
    for idx in range(54):
        face_i = idx // 9
        pos_in_face = idx % 9
        row, col = pos_in_face // 3, pos_in_face % 3
        r, g, b = int(data_rgb_f[idx][0]), int(data_rgb_f[idx][1]), int(data_rgb_f[idx][2])
        h, s, v = data_hsv[idx][0], data_hsv[idx][1], data_hsv[idx][2]
        gr = g / max(r, 1)
        label = face_label_map[face_i]
        is_center = "*" if pos_in_face == 4 else " "
        print(f"  [{idx:2d}] {label}({row},{col}){is_center} RGB=({r:3d},{g:3d},{b:3d}) HSV=({h:5.1f},{s:5.1f},{v:5.1f}) G/R={gr:.3f}")

    # 3. Per-sticker assignment
    sticker_costs = {}
    for idx in range(54):
        if idx in CENTER_IDXS:
            continue
        costs = {}
        for face_id in range(6):
            c = color_dist(data_hsv[idx], data_rgb_f[idx], face_id)
            costs[face_names[face_id]] = c
        sticker_costs[idx] = costs
        best_color = min(costs, key=costs.get)
        res[idx] = best_color

    # 4. Count-based auto-correction
    from collections import Counter
    counts = Counter(res)
    print(f"  Initial counts: {dict(counts)}")
    
    max_iterations = 20
    for iteration in range(max_iterations):
        counts = Counter(res)
        over_colors = [c for c in counts if counts[c] > 9]
        under_colors = [c for c in counts if counts[c] < 9]
        
        if not over_colors:
            break
        
        for over_c in over_colors:
            excess = counts[over_c] - 9
            candidates = []
            for idx in range(54):
                if idx in CENTER_IDXS:
                    continue
                if res[idx] == over_c and idx in sticker_costs:
                    costs = sticker_costs[idx]
                    sorted_costs = sorted(costs.items(), key=lambda x: x[1])
                    best_name, best_cost = sorted_costs[0]
                    for alt_name, alt_cost in sorted_costs[1:]:
                        if alt_name in under_colors:
                            margin = alt_cost - best_cost
                            candidates.append((margin, idx, alt_name))
                            break
            
            candidates.sort()
            for i in range(min(excess, len(candidates))):
                margin, idx, new_color = candidates[i]
                old_color = res[idx]
                res[idx] = new_color
                print(f"  Correction: [{idx}] {old_color} → {new_color} (margin={margin:.1f})")
                counts[old_color] -= 1
                counts[new_color] += 1
                under_colors = [c for c in counts if counts[c] < 9]

    # Debug: final assignment
    print("  === Final assignment ===")
    for idx in range(54):
        face_i = idx // 9
        pos_in_face = idx % 9
        row, col = pos_in_face // 3, pos_in_face % 3
        label = face_label_map[face_i]
        is_center = "*" if pos_in_face == 4 else " "
        print(f"  [{idx:2d}] {label}({row},{col}){is_center} → {res[idx]}")

    counts = Counter(res)
    print(f"  Final color counts: {dict(counts)}")


    return res, center_rgb, None


def generate_preview(image, points_dict, grid_colors):
    """
    Utility to draw the bounding boxes and sampled points on the image for the user to verify.
    points_dict: dict of face_name -> 4 points
    grid_colors: mapped colors to draw.
    """
    img_preview = image.copy()
    
    for face_name, points in points_dict.items():
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_preview, [pts], True, (0, 255, 0), 2)
        
        # We can also draw circles at the expected grid centers
        
    return img_preview

if __name__ == "__main__":
    pass
