
def diet_prolem():
    STRUCTURED_DATA = {
        "vars": {
            "title": "foods",
            "vars": {
                "chicken": {"cost": 3.0, "protein": 30, "carbs": 0, "fat": 4},
                "rice": {"cost": 0.5, "protein": 4, "carbs": 30, "fat": 0},
                "broccoli": {"cost": 1.0, "protein": 3, "carbs": 10, "fat": 1}
            }
        },
        "constraints": {
            "protein_min": 50,
            "carbs_max": 70,
            "fat_max": 10
        }
    }
    return {
        "title": "The Diet Problem",
        "description": f"""
                You need to choose a combination of foods to meet your daily nutritional requirements at the lowest cost.

                You can choose from:
                - Chicken Breast (${STRUCTURED_DATA["vars"]["vars"]["chicken"]["cost"]}/unit): {STRUCTURED_DATA["vars"]["vars"]["chicken"]["protein"]}g protein, {STRUCTURED_DATA["vars"]["vars"]["chicken"]["fat"]}g fat
                - Rice (${STRUCTURED_DATA["vars"]["vars"]["rice"]["cost"]}/unit): {STRUCTURED_DATA["vars"]["vars"]["rice"]["protein"]}g protein, {STRUCTURED_DATA["vars"]["vars"]["rice"]["carbs"]}g carbs
                - Broccoli (${STRUCTURED_DATA["vars"]["vars"]["broccoli"]["cost"]}/unit): {STRUCTURED_DATA["vars"]["vars"]["broccoli"]["protein"]}g protein, {STRUCTURED_DATA["vars"]["vars"]["broccoli"]["carbs"]}g carbs, {STRUCTURED_DATA["vars"]["vars"]["broccoli"]["fat"]}g fat

                Daily requirements:
                - At least {STRUCTURED_DATA["constraints"]["protein_min"]}g protein
                - At most {STRUCTURED_DATA["constraints"]["carbs_max"]}g carbs
                - At most {STRUCTURED_DATA["constraints"]["fat_max"]}g fat
            """,
        **STRUCTURED_DATA,
        "objective": "minimize_cost",
        "type": "linear"
    }
