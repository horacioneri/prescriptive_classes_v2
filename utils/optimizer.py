
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

def solve_diet_problem(foods, constraints):
    prob = LpProblem("Diet_Problem", LpMinimize)

    food_vars = {food: LpVariable(food, lowBound=0, cat='Continuous') for food in foods}

    prob += lpSum(food_vars[f] * foods[f]["cost"] for f in foods), "Total Cost"

    prob += lpSum(food_vars[f] * foods[f]["protein"] for f in foods) >= constraints["protein_min"], "ProteinRequirement"
    prob += lpSum(food_vars[f] * foods[f]["carbs"] for f in foods) <= constraints["carbs_max"], "CarbsLimit"
    prob += lpSum(food_vars[f] * foods[f]["fat"] for f in foods) <= constraints["fat_max"], "FatLimit"

    prob.solve()

    result = {
        "status": prob.status,
        "solution": {f: food_vars[f].varValue for f in foods},
        "objective": value(prob.objective)
    }

    return result
