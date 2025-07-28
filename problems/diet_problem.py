
PROBLEM = {
    "title": "The Diet Problem",
    "description": """
You need to choose a combination of foods to meet your daily nutritional requirements at the lowest cost.

You can choose from:
- Chicken Breast ($2/unit): 30g protein, 4g fat
- Rice ($0.5/unit): 3g protein, 30g carbs
- Broccoli ($1/unit): 2g protein, 10g carbs, 1g fat

Daily requirements:
- At least 50g protein
- At most 70g carbs
- At most 10g fat
    """,
    "foods": {
        "chicken": {"cost": 2.0, "protein": 30, "carbs": 0, "fat": 4},
        "rice": {"cost": 0.5, "protein": 3, "carbs": 30, "fat": 0},
        "broccoli": {"cost": 1.0, "protein": 2, "carbs": 10, "fat": 1}
    },
    "constraints": {
        "protein_min": 50,
        "carbs_max": 70,
        "fat_max": 10
    }
}
