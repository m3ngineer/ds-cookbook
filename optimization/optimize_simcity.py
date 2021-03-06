from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name="sim-city", sense=LpMaximize)

# Initialize the decision variables
s = LpVariable(name="seed", lowBound=0)
o = LpVariable(name="ore", lowBound=0)
m = LpVariable(name="metal", lowBound=0)
l = LpVariable(name="log", lowBound=0)
p = LpVariable(name="plastic", lowBound=0)
c = LpVariable(name="chemical", lowBound=0)
t = LpVariable(name="textile", lowBound=0)

# Seed: 30
# Ore: 40
# metal: 10
# log: 14
# plastic: 25
# chemical: 60
# textile: 90

# Add the objective function to the model
seed_discount = 0.1
ore_discount = 0
metal_discount = 0
log_discount = 0
plastic_discount = 0
chemical_discount = 0
textile_discount = 0
model += 30 * s * (1 - seed_discount) + 40 * o * (1 - ore_discount) + 10 * m * (1 - metal_discount) \
    + 14 * l * (1 - log_discount) + 25 * p * (1 - plastic_discount) + 60 * c * (1 - chemical_discount) \
    + 90 * t * (1 - textile_discount)

constraint = s + o + m + l + p + c + t <= 15

# Add the constraints to the model
click_penalty = 1.1 # penalty for every click that occurs
hours = 3
total_time = 3600 * hours
model += (constraint, "slot_constraint")
model += (s <= (total_time / 720) * click_penalty , "seed_constraint")
model += (o <= (total_time / 1800) * click_penalty , "ore_constraint")
model += (m <= (total_time / 60) * click_penalty , "metal_constraint")
model += (l <= (total_time / 180) * click_penalty , "log_constraint")
model += (p <= (total_time / 540) * click_penalty , "plastic_constraint")
model += (p <= (total_time / 7200) * click_penalty , "chemical_constraint")
model += (p <= (total_time / 10800) * click_penalty , "textile_constraint")

# Solve the problem
status = model.solve()

print(model)
print(status)
print(model.objective.value())

for var in model.variables():
    print(f"{var.name}: {var.value()}")
