import streamlit as st
import pandas as pd

def solution_evaluation(problem, user_vars):
    objective_function = 0
    constraints = {}
    constraints_met = False
    
    if problem["title"] == "The Diet Problem":
        protein = 0 
        carbs = 0
        fat = 0

        for var_name, attributes in problem["vars"]["vars"].items():
            quantity = user_vars.get(var_name, 0)
            objective_function += quantity * attributes["cost"]
            protein += quantity * attributes["protein"]
            carbs += quantity * attributes["carbs"]
            fat += quantity * attributes["fat"]
        
        constraints["protein_min"] = protein
        constraints["carbs_max"] = carbs
        constraints["fat_max"] = fat

        if (protein >= problem["constraints"]["protein_min"] and
            carbs <= problem["constraints"]["carbs_max"] and
            fat <= problem["constraints"]["fat_max"]):
            constraints_met = True

        return objective_function, constraints, constraints_met


def diet_problem():
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
            
def food_distribution_problem():
    STRUCTURED_DATA = {
        "vars": {
            "title": "Country",
            "vars": {
                "Democratic Republic of the Congo": {"Packages needs": 10000.0, "Distribution cost": 0.8, "Population served": 500},
                "South Sudan": {"Packages needs": 30000.0, "Distribution cost": 0.8, "Population served": 500},
                "Central African Republic": {"Packages needs": 50000.0, "Distribution cost": 0.8, "Population served": 500},
                "Syria": {"Packages needs": 45000.0, "Distribution cost": 0.53, "Population served": 400},
                "Yemen": {"Packages needs": 14000.0, "Distribution cost": 0.53, "Population served": 400},
                "Myanmar": {"Packages needs": 23000.0, "Distribution cost": 0.67, "Population served": 350},
                "Bangladesh": {"Packages needs": 29000.0, "Distribution cost": 0.67, "Population served": 350}
            }
        },
        "constraints": {
            "budget_max": 100000,
            "food_packages_max": 170000,
        }
    }
    # Access the nested data
    country_data = STRUCTURED_DATA["vars"]["vars"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(country_data, orient='index')

    # Optionally reset index to make 'Country' a column
    df = df.reset_index().rename(columns={"index": "Country"})

    return {
        "title": "The Food Distribution Problem",
        "description": f"""
                A food distribution program assists countries in emergency.
                On a weekly basis a total of 170 000 food packages are available to sent to a set
                of countries needing help. The following table identifies the packages needs per
                country and the unitary distribution cost.

                {st.dataframe(df)}

                The total available budget for the distribution amounts to 100 000€. How
                should the food packages be distributed to maximize the total population
                served?
            """,
        **STRUCTURED_DATA,
        "objective": "minimize_cost",
        "type": "linear"
    }