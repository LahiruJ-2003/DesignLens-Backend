import torch
from schemas import DesignPayload
from typing import Tuple

# Maps every element type the system can encounter to a 0-1 value.
# Groups: 0=rect/button, 0.125=circle, 0.25=text, 0.375=image,
#         0.5=line,       0.625=frame/container, 0.75=triangle, 1.0=star/arrow/other
TYPE_MAP: dict[str, float] = {
    # Frontend app types
    'rectangle': 0.0,   'rect': 0.0,
    'circle': 0.125,    'ellipse': 0.125,
    'text': 0.25,       'Text': 0.25,
    'image': 0.375,
    'line': 0.5,
    'frame': 0.625,
    'triangle': 0.75,
    'star': 0.875,
    'arrow': 1.0,
    # Synthetic dataset legacy types
    'Shape': 0.0,
    # RICO Android class names (already split on '.' by the parser)
    'Button': 0.0,      'ImageButton': 0.0,   'CheckBox': 0.0,
    'EditText': 0.0,    'Switch': 0.0,        'ToggleButton': 0.0,
    'RadioButton': 0.125,
    'TextView': 0.25,   'EditText': 0.25,
    'ImageView': 0.375,
    'View': 0.5,        'ProgressBar': 0.5,   'SeekBar': 0.5,
    'LinearLayout': 0.625,  'RelativeLayout': 0.625, 'FrameLayout': 0.625,
    'ConstraintLayout': 0.625, 'ScrollView': 0.625,   'RecyclerView': 0.625,
    'ListView': 0.625,  'GridView': 0.625,    'CardView': 0.625,
}


def compute_distance(el1, el2) -> float:
    cx1 = el1.x + el1.width / 2
    cy1 = el1.y + el1.height / 2
    cx2 = el2.x + el2.width / 2
    cy2 = el2.y + el2.height / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def extract_color_features(hex_color: str) -> list[float]:
    """Return [R, G, B] normalised to 0-1. Falls back to [0, 0, 0] on bad input."""
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
    """
    Convert a DesignPayload into PyTorch tensors (node features, edge index).

    Node feature vector (8 values per element):
      [x/fw, y/fh, w/fw, h/fh, R, G, B, type_encoded]

    Coordinates are normalised by the frame dimensions so the model learns
    position semantics (e.g. 0.5 = centred) rather than raw pixel values.
    Previously coordinates were divided by a fixed 1000 which made a centred
    element on a 390px phone look the same as a left-edge element on a 1440px
    desktop canvas.
    """
    fw = payload.frame_width or 390.0
    fh = payload.frame_height or 844.0

    node_features = []
    for el in payload.elements:
        rgb = extract_color_features(el.fill)
        type_val = TYPE_MAP.get(el.type, 0.0)

        features = [
            el.x / fw,          # relative x position  (was x/1000)
            el.y / fh,          # relative y position  (was y/1000)
            el.width / fw,      # relative width       (was w/1000)
            el.height / fh,     # relative height      (was h/1000)
            rgb[0],             # R
            rgb[1],             # G
            rgb[2],             # B  (was dropped before — "quick hack")
            type_val,           # element type encoding
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge threshold: 30% of frame width so spatial proximity is relative to
    # screen size. Previously fixed at 300px which connected nearly every element
    # on a 390px phone screen but barely anything on a 1440px desktop.
    edge_threshold = fw * 0.30
    edges = []
    for i, el1 in enumerate(payload.elements):
        for j, el2 in enumerate(payload.elements):
            if i != j and compute_distance(el1, el2) < edge_threshold:
                edges.append([i, j])

    if len(edges) == 0:
        for i in range(len(payload.elements)):
            edges.append([i, i])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index
