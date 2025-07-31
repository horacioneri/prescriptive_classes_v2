import streamlit as st
import pandas as pd

def evaluation_printing(objective, constraints, constraints_met, problem):
    st.markdown("### Evaluation")
    st.metric("Objective", f"{objective:.2f}")

    constraints_string = ""
    for constraint_name, actual in constraints.items():
        if constraint_name in problem["constraints"]:
            limit = problem["constraints"][constraint_name]
            constraints_string += f"{constraint_name}: {actual} (limit: {limit})  |  "
        else:
            constraints_string += f"{actual}  |  "

    st.write(constraints_string)

    if constraints_met:
        st.success("Constraints met!")
    else:
        st.warning("Constraints not satisfied.")

def solution_evaluation(problem, user_vars):
    objective_function = 0
    constraints = {}
    constraints_met = False
    
    if problem["title"] == "The Diet Problem":
        
        foods = problem["vars"]["vars"]
        # Initialize totals
        totals = {"protein": 0, "carbs": 0, "fat": 0, "sodium": 0, "fiber": 0}

        for food, qty in user_vars.items():
            food_data = foods[food]
            for nutrient in totals:
                totals[nutrient] += food_data.get(nutrient, 0) * qty
            objective_function += food_data.get("cost", 0) * qty

        
        constraints["protein_min"] = totals["protein"]
        constraints["carbs_max"] = totals["carbs"]
        constraints["carbs_min"] = totals["carbs"]
        constraints["fat_max"] = totals["fat"]
        constraints["sodium_max"] = totals["sodium"]
        constraints["fiber_min"] = totals["fiber"]
        constraints["food_diversity_min"] = sum(1 for v in user_vars.values() if v >= 1)

        if (constraints["protein_min"] >= problem["constraints"]["protein_min"] and
            constraints["carbs_max"] <= problem["constraints"]["carbs_max"] and
            constraints["carbs_min"] >= problem["constraints"]["carbs_min"] and
            constraints["fat_max"] <= problem["constraints"]["fat_max"] and
            constraints["sodium_max"] <= problem["constraints"]["sodium_max"] and
            constraints["fiber_min"] >= problem["constraints"]["fiber_min"] and
            constraints["food_diversity_min"] >= problem["constraints"]["food_diversity_min"]
            ):
            constraints_met = True
    
    elif problem["title"] == "The Food Distribution Problem":
        budget = 0
        packages = 0
        too_much = False
        other_string = ""

        for var_name, attributes in problem["vars"]["vars"].items():
            quantity = user_vars.get(var_name, 0)
            objective_function += quantity * attributes["Population served per package"]
            budget += quantity * attributes["Distribution cost per package"]
            packages += quantity
            if quantity > attributes["Packages needs"]:
                too_much = True
                other_string += var_name + " is receiving more quantity than needed, "

        constraints["budget_max"] = budget
        constraints["food_packages_max"] = packages
        constraints["other"] = other_string
    
        if (budget <= problem["constraints"]["budget_max"] and
            packages <= problem["constraints"]["food_packages_max"] and
            not too_much):
            constraints_met = True

    return objective_function, constraints, constraints_met


def diet_problem():
    STRUCTURED_DATA = {
        "vars": {
            "vars": {
            "Chicken Breast": {"cost": 2.5, "protein": 30, "fat": 3, "carbs": 0, "sodium": 70, "fiber": 0},
            "Tofu": {"cost": 1.5, "protein": 10, "fat": 5, "carbs": 3, "sodium": 15, "fiber": 2},
            "Rice": {"cost": 0.5, "protein": 3, "fat": 1, "carbs": 40, "sodium": 0, "fiber": 1},
            "Broccoli": {"cost": 0.7, "protein": 2.5, "fat": 0.5, "carbs": 6, "sodium": 30, "fiber": 2.5},
            "Cheese": {"cost": 1.2, "protein": 6, "fat": 9, "carbs": 1, "sodium": 180, "fiber": 0},
            "Avocado": {"cost": 1.3, "protein": 2, "fat": 15, "carbs": 9, "sodium": 10, "fiber": 7},
            "Oats": {"cost": 0.8, "protein": 5, "fat": 3, "carbs": 27, "sodium": 0, "fiber": 4}
            }
        },
        "constraints": {
            "protein_min": 50,
            "carbs_min": 130,
            "carbs_max": 300,
            "fat_max": 70,
            "sodium_max": 2000,
            "fiber_min": 25,
            "food_diversity_min": 3,
        }
    }
    # Access the nested data
    food_data = STRUCTURED_DATA["vars"]["vars"]

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(food_data, orient='index')

    # Optionally reset index to make 'Country' a column
    df = df.reset_index().rename(columns={"index": "Food"})
    return {
        "title": "The Diet Problem",
        "description": f"""
                Design a meal plan that meets nutritional needs at the lowest cost. Balance your food choices to avoid too much fat or sodium, and ensure fiber and variety.
                Find below the options available and their characteristics. 

                Daily requirements:
                - At least {STRUCTURED_DATA["constraints"]["protein_min"]}g protein
                - At least {STRUCTURED_DATA["constraints"]["carbs_min"]}g carbs
                - At most {STRUCTURED_DATA["constraints"]["carbs_max"]}g carbs
                - At most {STRUCTURED_DATA["constraints"]["fat_max"]}g fat
                - At most {STRUCTURED_DATA["constraints"]["sodium_max"]}g sodium
                - At least {STRUCTURED_DATA["constraints"]["fiber_min"]}g fiber
                - At least 1 unit of at least {STRUCTURED_DATA["constraints"]["food_diversity_min"]} different types of food

            """,
        "dataframe": df,
        **STRUCTURED_DATA,
        "structured_data": STRUCTURED_DATA,
        "objective": "minimize_cost",
        "type": "linear"
    }
            
def food_distribution_problem():
    STRUCTURED_DATA = {
        "vars": {
            "title": "Country",
            "vars": {
                "Democratic Republic of the Congo": {"Packages needs": 10000.0, "Distribution cost per package": 0.8, "Population served per package": 0.500},
                "South Sudan": {"Packages needs": 30000.0, "Distribution cost per package": 0.8, "Population served per package": 0.500},
                "Central African Republic": {"Packages needs": 50000.0, "Distribution cost per package": 0.8, "Population served per package": 0.500},
                "Syria": {"Packages needs": 45000.0, "Distribution cost per package": 0.53, "Population served per package": 0.400},
                "Yemen": {"Packages needs": 14000.0, "Distribution cost per package": 0.53, "Population served per package": 0.400},
                "Myanmar": {"Packages needs": 23000.0, "Distribution cost per package": 0.67, "Population served per package": 0.350},
                "Bangladesh": {"Packages needs": 29000.0, "Distribution cost per package": 0.67, "Population served per package": 0.350}
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
                of countries needing help. The table below, identifies the packages needs per
                country and the unitary distribution cost.

                The total available budget for the distribution amounts to 100 000€. How
                should the food packages be distributed to maximize the total population
                served?
            """,
        "dataframe": df,
        **STRUCTURED_DATA,
        "structured_data": STRUCTURED_DATA,
        "objective": "maximize_population_served",
        "type": "linear"
    }