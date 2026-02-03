from typing import Any, Dict, List, Tuple

from core.contracts import RetrievedChunk


def build_annotations(chunks: List[RetrievedChunk], color: str = "red") -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    for chunk in chunks:
        for polygon_entry in chunk.polygons:
            page_number = int(polygon_entry.get("page_number", 1))
            polygon = polygon_entry.get("polygon", [])
            bbox = _bbox_from_polygon(polygon)
            if not bbox:
                continue
            annotations.append(
                {
                    "page": page_number,
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                    "color": color,
                }
            )
    return annotations


def build_annotations_with_index(
    chunks: List[RetrievedChunk], color: str = "red"
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    annotations: List[Dict[str, Any]] = []
    index_map: Dict[str, int] = {}
    for chunk in chunks:
        first_index = None
        for polygon_entry in chunk.polygons:
            page_number = int(polygon_entry.get("page_number", 1))
            polygon = polygon_entry.get("polygon", [])
            bbox = _bbox_from_polygon(polygon)
            if not bbox:
                continue
            annotations.append(
                {
                    "page": page_number,
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                    "color": color,
                }
            )
            if first_index is None:
                first_index = len(annotations)
        if first_index is not None:
            index_map[chunk.chunk_id] = first_index
    return annotations, index_map


def _bbox_from_polygon(polygon: List[Dict[str, Any]]) -> Dict[str, float]:
    if not polygon:
        return {}
    xs = [float(point["x"]) for point in polygon if "x" in point]
    ys = [float(point["y"]) for point in polygon if "y" in point]
    if not xs or not ys:
        return {}
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return {"x": min_x, "y": min_y, "width": max_x - min_x, "height": max_y - min_y}
