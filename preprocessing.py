import torch
from schemas import DesignPayload
from typing import Tuple

# Each element type is mapped to a number between 0 and 1
# so the neural network can understand what kind of element it is.
# I grouped similar types together (buttons/rectangles = 0.0, text = 0.25, images = 0.375, etc.)
TYPE_MAP: dict[str, float] = {
    # Types used in the frontend canvas
    'rectangle': 0.0,   'rect': 0.0,
    'circle': 0.125,    'ellipse': 0.125,
    'text': 0.25,       'Text': 0.25,
    'image': 0.375,
    'line': 0.5,
    'frame': 0.625,
    'triangle': 0.75,
    'star': 0.875,
    'arrow': 1.0,
    # Types used in the synthetic dataset
    'Shape': 0.0,
    # Android class names from the RICO dataset
    'Button': 0.0,      'ImageButton': 0.0,   'CheckBox': 0.0,
    'EditText': 0.0,    'Switch': 0.0,        'ToggleButton': 0.0,
    'RadioButton': 0.125,
    'TextView': 0.25,
    'ImageView': 0.375,
    'View': 0.5,        'ProgressBar': 0.5,   'SeekBar': 0.5,
    'LinearLayout': 0.625,  'RelativeLayout': 0.625, 'FrameLayout': 0.625,
    'ConstraintLayout': 0.625, 'ScrollView': 0.625,   'RecyclerView': 0.625,
    'ListView': 0.625,  'GridView': 0.625,    'CardView': 0.625,
}


def compute_distance(el1, el2) -> float:
    # Calculate the distance between the centre points of two elements
    # Used to decide whether two elements should be connected by an edge in the graph
    cx1 = el1.x + el1.width / 2
    cy1 = el1.y + el1.height / 2
    cx2 = el2.x + el2.width / 2
    cy2 = el2.y + el2.height / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def extract_color_features(hex_color: str) -> list[float]:
    # Convert a hex colour like #3b82f6 into three numbers [R, G, B] between 0 and 1
    # The model uses colour as part of the node features to detect palette consistency
    if not hex_color or hex_color.startswith('rgb') or hex_color.startswith('var'):
        return [0.0, 0.0, 0.0]
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return [r, g, b]
    return [0.0, 0.0, 0.0]


def payload_to_graph(payload: DesignPayload) -> Tuple[torch.Tensor, torch.Tensor]:
    # This is the main preprocessing function.
    # It takes the list of UI elements from the frontend and converts them
    # into a graph format that PyTorch Geometric can process.
    #
    # Each element becomes a NODE with 8 features:
    #   [x/fw, y/fh, w/fw, h/fh, R, G, B, type]
    #
    # I divide positions by the frame size (not a fixed 1000) so that
    # a centred element always has x=0.5 regardless of screen size.

    fw = payload.frame_width or 390.0
    fh = payload.frame_height or 844.0

    node_features = []
    for el in payload.elements:
        rgb = extract_color_features(el.fill)
        type_val = TYPE_MAP.get(el.type, 0.0)

        features = [
            el.x / fw,       # normalised x position (0 = left edge, 1 = right edge)
            el.y / fh,       # normalised y position (0 = top, 1 = bottom)
            el.width / fw,   # normalised width
            el.height / fh,  # normalised height
            rgb[0],          # red channel
            rgb[1],          # green channel
            rgb[2],          # blue channel
            type_val,        # element type as a number
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Two elements get connected by an EDGE if they are close to each other.
    # I use 30% of the frame width as the threshold so it scales with screen size.
    # Close elements affect each other's score in the graph attention network.
    edge_threshold = fw * 0.30
    edges = []
    for i, el1 in enumerate(payload.elements):
        for j, el2 in enumerate(payload.elements):
            if i != j and compute_distance(el1, el2) < edge_threshold:
                edges.append([i, j])

    # If no edges were created (single element or everything too far apart),
    # add self-loops so the model still has something to process
    if len(edges) == 0:
        for i in range(len(payload.elements)):
            edges.append([i, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index
