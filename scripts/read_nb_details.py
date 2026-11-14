import json

with open("notebooks/capstone.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    print(f"=== CELL {i+1} ({c['cell_type']}) ===")
    print(src[:400])
    print("-" * 50)
