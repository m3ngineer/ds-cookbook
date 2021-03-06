from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name="sim-city", sense=LpMaximize)

# Initialize the decision variables
s = LpVariable(name="seed", lowBound=0)
o = LpVariable(name="ore", lowBound=0)
m = LpVariable(name="metal", lowBound=0)
l = LpVariable(name="log", lowBound=0)
p = LpVariable(name="plastic", lowBound=0)

# Seed: 30
# Ore: 40
# metal: 10
# log: 14
# plastic: 25

# Add the objective function to the model
model += 30 * s + 40 * o + 10 * m + 14 * l + 25 * p

constraint = s + o + m + l + p <= 15

# Add the constraints to the model
delay = 1
model += (constraint, "slot_constraint")
model += (s <= 3600 / (720 * delay) , "seed_constraint")
model += (o <= 3600 / (1800 * delay) , "ore_constraint")
model += (m <= 3600 / (60 * delay) , "metal_constraint")
model += (l <= 3600 / (180 * delay) , "log_constraint")
model += (p <= 3600 / (540 * delay) , "plastic_constraint")

# Solve the problem
status = model.solve()

print(model)
print(status)
print(model.objective.value())

for var in model.variables():
    print(f"{var.name}: {var.value()}")
