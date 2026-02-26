import kociemba

def solve_cube(cube_string: str) -> str:
    """
    Solve a Rubik's cube from a string representation.
    The string should be 54 characters long, consisting of 
    the letters U, R, F, D, L, B corresponding to the faces.
    The order of the 54 characters is U1..U9, R1..R9, F1..F9, D1..D9, L1..L9, B1..B9.
    Reading order is row by row, left to right, top to bottom for each face.
    """
    if len(cube_string) != 54:
        raise ValueError(f"Cube string must be exactly 54 characters. Got {len(cube_string)}.")
    
    try:
        # kociemba.solve returns a string like "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
        solution = kociemba.solve(cube_string)
        return solution
    except ValueError as e:
        # Kociemba will raise ValueError for invalid cube configurations
        raise ValueError(f"Invalid cube configuration: {str(e)}")
    except Exception as e:
        raise Exception(f"Error solving cube: {str(e)}")

# Define the face order expected by Kociemba
KOCIEMBA_FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

def map_colors_to_string(faces_colors: dict) -> str:
    """
    Map a dictionary of face colors to a Kociemba standard string.
    faces_colors should be a dict where keys are face names (U, R, F, D, L, B)
    or color names (e.g., 'white', 'red') and values are 9-element lists of colors.
    
    This function discovers which color represents which face by looking at the center
    piece of each face (index 4 in the 9-element list).
    """
    if len(faces_colors) != 6:
        raise ValueError("Must provide exactly 6 faces.")
    
    # Identify which color represents which face based on the center piece
    color_to_face = {}
    
    # Try using the standard keys if they exist
    if all(face in faces_colors for face in KOCIEMBA_FACE_ORDER):
        faces_to_process = {f: faces_colors[f] for f in KOCIEMBA_FACE_ORDER}
    else:
        # Otherwise, we need to intuitively figure out which is which, 
        # which usually means the input should already be keyed by the logical face U, R, etc.
        # If not, this logic requires a specific dictionary structure.
        raise ValueError("Input dictionary must contain keys: 'U', 'R', 'F', 'D', 'L', 'B'")

    for face_name in KOCIEMBA_FACE_ORDER:
        face_colors = faces_to_process[face_name]
        if len(face_colors) != 9:
            raise ValueError(f"Face {face_name} must have exactly 9 colors.")
        center_color = face_colors[4]
        color_to_face[center_color] = face_name
        
    if len(color_to_face) != 6:
        raise ValueError("Cube does not have 6 distinct center colors. Invalid configuration.")

    # Now build the string
    cube_string = []
    for face_name in KOCIEMBA_FACE_ORDER:
        face_colors = faces_to_process[face_name]
        for color in face_colors:
            if color not in color_to_face:
                raise ValueError(f"Found color '{color}' that is not any center color.")
            cube_string.append(color_to_face[color])
            
    return "".join(cube_string)

if __name__ == "__main__":
    # Test valid cube
    solved_cube = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    print(f"Solved cube string: {solved_cube}")
    
    # A simple scramble: R
    # U right column becomes F right column
    # F right column becomes D right column
    # D right column becomes B left column (reversed)
    # B left column (reversed) becomes U right column
    # R rotates clockwise
    
    try:
        # Just test that the import works for now
        res = kociemba.solve(solved_cube)
        print(f"Solution for solved cube: '{res}'")
    except Exception as e:
        print(f"Error: {e}")
