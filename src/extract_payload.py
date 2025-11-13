# extract_payload.py
# Extract on-wire encrypted payloads from tshark JSON and output as base64 dataset.

import json, re, base64, os
from pathlib import Path
from typing import Any, Optional, List, Dict

def normalize_hex_to_bytes(hex_field: Any) -> Optional[bytes]:
    """Convert tshark hex fields (string or list, possibly colon-separated) into bytes."""
    if hex_field is None:
        return None
    if isinstance(hex_field, list):
        for item in hex_field:
            b = normalize_hex_to_bytes(item)
            if b:
                return b
        return None
    if not isinstance(hex_field, str):
        return None
    s = hex_field.strip().replace(":", "").replace(" ", "").replace("\n", "").lower()
    if not s or (len(s) % 2 != 0) or re.search(r"[^0-9a-f]", s):
        return None
    try:
        return bytes.fromhex(s)
    except Exception:
        return None

def extract_onwire_payload(frame: Dict) -> Optional[bytes]:
    """Priority: udp.payload > tcp.payload > data.data (extract only encrypted on-wire payload)."""
    layers = frame.get("_source", {}).get("layers", {})
    udp = layers.get("udp")
    if isinstance(udp, dict) and "udp.payload" in udp:
        b = normalize_hex_to_bytes(udp.get("udp.payload"))
        if b:
            return b
    tcp = layers.get("tcp")
    if isinstance(tcp, dict) and "tcp.payload" in tcp:
        b = normalize_hex_to_bytes(tcp.get("tcp.payload"))
        if b:
            return b
    data_layer = layers.get("data")
    if isinstance(data_layer, dict) and "data.data" in data_layer:
        b = normalize_hex_to_bytes(data_layer.get("data.data"))
        if b:
            return b
    return None

def process_file(in_json_path: str, out_json_path: str, label: str):
    IN_JSON = Path(in_json_path)
    OUT_JSON = Path(out_json_path)

    if not IN_JSON.exists():
        raise FileNotFoundError(f"Input file not found: {IN_JSON}")

    with IN_JSON.open("r", encoding="utf-8") as f:
        content = json.load(f)

    if not isinstance(content, list):
        raise ValueError("Input is not a standard tshark JSON (top level must be an array of frames).")

    b64_list: List[str] = []
    for frame in content:
        b = extract_onwire_payload(frame)
        if b:
            b64_list.append(base64.b64encode(b).decode("ascii"))

    out_obj = {label: b64_list}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Done: extracted {len(b64_list)} payloads -> {OUT_JSON}")

if __name__ == "__main__":
    # Example usage (modify paths and label as needed)
    process_file("chatgpt1.json", "data1.json", "chatgpt")
