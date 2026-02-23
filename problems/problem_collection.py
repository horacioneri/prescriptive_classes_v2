import pandas as pd
from utils.route_utils import build_route_ids

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

    elif problem["title"] == "The Europe Traveling Route Problem":
        cities = problem["vars"]["cities"]
        distance_matrix = problem["vars"]["distance_matrix_km"]

        route = build_route_ids(user_vars, problem)
        # ensure tour closes back to start for evaluation
        if route and route[0] != route[-1]:
            route.append(route[0])

        if len(route) < 2:
            constraints.update({
                "visited_cities_min": len(set(route)),
                "duplicates_max": 0,
                "return_to_start_min": 0,
                "start_city_min": 0,
            })
            constraints_met = False
            return float("inf"), constraints, constraints_met

        closed = route[0] == route[-1]
        route_core = route[:-1] if closed else route

        unique_cities = len(set(route_core))
        duplicates = len(route_core) - unique_cities

        # Sum distances along the provided path (includes closing leg if supplied)
        for i in range(len(route) - 1):
            objective_function += distance_matrix[route[i]][route[i + 1]]

        constraints["visited_cities_min"] = unique_cities
        constraints["duplicates_max"] = duplicates
        constraints["return_to_start_min"] = 1 if closed else 0

        start_id = problem.get("constraints", {}).get("start_city_id")
        start_ok = start_id is None or route[0] == start_id
        constraints["start_city_min"] = 1 if start_ok else 0

        constraints_met = (
            unique_cities >= problem["constraints"]["visited_cities_min"]
            and duplicates <= problem["constraints"]["duplicates_max"]
            and constraints["return_to_start_min"] >= problem["constraints"]["return_to_start_min"]
            and constraints["start_city_min"] >= problem["constraints"]["start_city_min"]
        )


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

def europe_traveling_route():
    STRUCTURED_DATA = {
        "vars": {
            "title": "Cities",
            "cities": [
                {"id": 0, "name": "Porto", "latitude": 41.1579, "longitude": -8.6291},
                {"id": 1, "name": "London", "latitude": 51.5074, "longitude": -0.1278},
                {"id": 2, "name": "Paris", "latitude": 48.8566, "longitude": 2.3522},
                {"id": 3, "name": "Madrid", "latitude": 40.4168, "longitude": -3.7038},
                {"id": 4, "name": "Berlin", "latitude": 52.5200, "longitude": 13.4050},
                {"id": 5, "name": "Rome", "latitude": 41.9028, "longitude": 12.4964},
                {"id": 6, "name": "Amsterdam", "latitude": 52.3676, "longitude": 4.9041},
                {"id": 7, "name": "Barcelona", "latitude": 41.3851, "longitude": 2.1734},
                {"id": 8, "name": "Brussels", "latitude": 50.8503, "longitude": 4.3517},
                {"id": 9, "name": "Vienna", "latitude": 48.2082, "longitude": 16.3738},
                {"id": 10, "name": "Lisbon", "latitude": 38.7223, "longitude": -9.1393},
                {"id": 11, "name": "Prague", "latitude": 50.0755, "longitude": 14.4378},
                {"id": 12, "name": "Munich", "latitude": 48.1351, "longitude": 11.5820},
                {"id": 13, "name": "Milan", "latitude": 45.4642, "longitude": 9.1900}
            ],
            "distance_matrix_km": [
                [0,1321,1213,423,2085,1756,1612,903,1468,2115,274,2038,1770,1515],
                [1321,0,344,1263,932,1434,358,1139,321,1235,1586,1038,919,1183],
                [1213,344,0,1053,877,1105,430,831,264,1033,1456,1034,878,637],
                [423,1263,1053,0,1869,1364,1481,505,1317,1810,502,1864,1413,1189],
                [2085,932,877,1869,0,1183,576,1499,651,524,2319,280,504,846],
                [1756,1434,1105,1364,1183,0,1296,859,1173,764,1776,1185,681,590],
                [1612,358,430,1481,576,1296,0,1238,173,935,1850,897,636,889],
                [903,1139,831,505,1499,859,1238,0,1066,1350,1166,1418,1213,772],
                [1468,321,264,1317,651,1173,173,1066,0,915,1680,867,690,741],
                [2115,1235,1033,1810,524,764,935,1350,915,0,2358,676,598,780],
                [274,1586,1456,502,2319,1776,1850,1166,1680,2358,0,2167,1876,1706],
                [2038,1038,1034,1864,280,1185,897,1418,867,676,2167,0,296,657],
                [1770,919,878,1413,504,681,636,1213,690,598,1876,296,0,385],
                [1515,1183,637,1189,846,590,889,772,741,780,1706,657,385,0]
            ]
        },
    }
    # Access the nested data
    city_data = STRUCTURED_DATA["vars"]["cities"]
    distance_matrix = STRUCTURED_DATA["vars"]["distance_matrix_km"]

    # Extract city names in correct order
    city_names = [c["name"] for c in city_data]
    
    # Build base dataframe (city + coordinates)
    wide_df = pd.DataFrame({
        "city": city_names,
        "latitude": [c["latitude"] for c in city_data],
        "longitude": [c["longitude"] for c in city_data]
    })
    
    # Add distance columns
    for i, city in enumerate(city_names):
        wide_df[f"dist_to_{city}"] = [row[i] for row in distance_matrix]

    return {
        "title": "The Europe Traveling Route Problem",
        "description": f"""
                An australian man is traveling to Europe and wants to visit all the selected 13 cities in the shortest distance possible. 
                The flight from Sydney and the flight back to Sydney are both to/from the same airport.
                
                In what ordet should he visit the cities?
            """,
        "dataframe": wide_df,
        **STRUCTURED_DATA,
        "structured_data": STRUCTURED_DATA,
        "objective": "minimize_distance",
        "type": "linear",
        "constraints": {
            "visited_cities_min": len(city_names),
            "duplicates_max": 0,
            "return_to_start_min": 1,   # require tour to close
            "start_city_min": 0         # starting city is free
        }
    }
