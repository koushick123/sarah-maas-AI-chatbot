import json

json_data = """
{
  "book_title": "Sarah Maas - Assassins Blade",
  "feminist_analysis": {
    "female_agency": [
      {
        "character": "Celaena Sardothien",
        "description": "Exhibits agency by ignoring the young man's snap and focusing on studying the other people in the room, despite feeling exhausted."
      }
    ],
    
    "intersectionality": [
      {
        "character": "Aelin Galathynius",
        "analysis": "Navigates both political leadership and personal trauma, emphasizing layered identities."
      }
    ],
    "oppressive_structures": [
      {
        "scene": "Maeve’s control over information and power",
        "analysis": "Represents a matriarchal oppression mirroring patriarchal control mechanisms."
      }
    ]
  }
}
"""

# Parse JSON
data = json.loads(json_data)

# Extract only present feminist_analysis fields
present_fields = {}
if "feminist_analysis" in data:
    for key, value in data["feminist_analysis"].items():
        if value:  # only include non-empty fields
            present_fields[key] = value

# Optional: include book title
result = {
    "book_title": data.get("book_title"),
    "feminist_analysis": present_fields
}

# Print the result
# print(json.dumps(result, indent=2, ensure_ascii=False))
result.pop("feminist_analysis")
print(json.dumps(result))