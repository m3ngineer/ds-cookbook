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
h = LpVariable(name="hammer", lowBound=0)
mt = LpVariable(name="measuring tape", lowBound=0)
sh = LpVariable(name="shovel", lowBound=0)
pl = LpVariable(name="plank", lowBound=0)
br = LpVariable(name="brick", lowBound=0)

# Seed: 30
# Ore: 40
# metal: 10
# log: 14
# plastic: 25
# chemical: 60
# textile: 90
# hammer: 90, 1 log, 1 metal 14 min
# measuring tape: 110, 1 metal, 1 plastic, 20 min
# shovel: 150, 1 metal, 1 log, 1 plastic, 30 time
# chair: 2 log, 1 nail, 1  hammer
# table: 1 plank, 2 nail, 1 hammer
# leek: 2 seed
# plank: 120, 2 log, 30 min
# brick: 190, 2 ore, 20 min
# nail: 80, 2 metal, 5 min

materials = {
    'nail': {'material': [('metal', 2)], 'time': 300, 'revenue': 80, 'discount': 0},
    'brick': {'material': [('mineral', 2)], 'time': 1200, 'revenue': 190, 'discount': 0},
    'plank': {'material': [('log', 2)], 'time': 1800, 'revenue': 120, 'discount': 0},
    'leek': {'material': [('seed', 2)], 'time': 1200, 'revenue': 160, 'discount': 0},
    # 'table': {'material': [('plank', 1), ('nail', 2), ('hammer', 1)], 'time': 1800, 'revenue': 80},
    'seed': {'material': [], 'time': 720, 'revenue': 30, 'discount': 0},
    'mineral': {'material': [], 'time': 1800, 'revenue': 40, 'discount': 0},
    'log': {'material': [], 'time': 180, 'revenue': 14, 'discount': 0},
    'plastic': {'material': [], 'time': 540, 'revenue': 25, 'discount': 0},
    'chemical': {'material': [], 'time': 7200, 'revenue': 60, 'discount': 0},
    'textile': {'material': [], 'time': 10800, 'revenue': 90, 'discount': 0},
    'hammer': {'material': [('log', 1), ('metal', 1)], 'time': 840, 'revenue': 90, 'discount': 0},
    'measuring_tape': {'material': [('log', 1), ('plastic', 1)], 'time': 1200, 'revenue': 110, 'discount': 0},
    'shovel': {'material': [('metal', 1), ('plastic', 1), ('log', 1)], 'time': 1800, 'revenue': 150, 'discount': 0},
}

# Add the objective function to the model
seed_discount = 0.1
ore_discount = 0
metal_discount = 0
log_discount = 0
plastic_discount = 0
chemical_discount = 0
textile_discount = 0
hammer_discount = 0
measuring_tape_discount = 0
shovel_discount = 0
brick_discount = 0
plank_discount = 0
model += 30 * s * (1 - seed_discount) + 40 * o * (1 - ore_discount) + 10 * m * (1 - metal_discount) \
    + 14 * l * (1 - log_discount) + 25 * p * (1 - plastic_discount) + 60 * c * (1 - chemical_discount) \
    + 90 * t * (1 - textile_discount) + 90 * h * (1 - hammer_discount) \
    + 110 * mt * (1 - measuring_tape_discount) + 150 * sh * (1 - shovel_discount) \
    + 120 * pl * (1 - plank_discount) + 190 * br * (1 - brick_discount)

expr = None
for k,v in materials.items():
    print(k)
    expr += v['revenue'] * LpVariable(name=k, lowBound=0) * (1-v['discount'])

print(expr)
# model += expr

total_factory_slots = 15
constraint = s + o + m + l + p + c + t <= total_factory_slots

# Add the constraints to the model
click_penalty = 1.1 # penalty for every click that occurs
hours = 3
total_time = 3600 * hours
model += (constraint, "slot_constraint")
model += (s * 720 <= (total_time) * click_penalty , "seed_constraint")
model += (o * 1800 <= (total_time) * click_penalty , "ore_constraint")
model += (m * 60 <= (total_time) * click_penalty , "metal_constraint")
model += (l * 180 <= (total_time) * click_penalty , "log_constraint")
model += (p * 540 <= (total_time) * click_penalty , "plastic_constraint")
model += (c * 7200 <= (total_time) * click_penalty , "chemical_constraint")
model += (t * 10800<= (total_time) * click_penalty , "textile_constraint")
model += (h * (60 + 180 + 840) * click_penalty <= total_time, "hammer_constraint")
model += (mt * (60 + 160 + 1200) * click_penalty <= total_time, "measuring_tape_constraint")
model += (sh * (60 + 180 + 540 + 1800) * click_penalty <= total_time, "shovel_constraint")
model += (pl * (180 + 1800) * click_penalty <= total_time, "plank_constraint")
model += (br * (1800 + 1200) * click_penalty <= total_time, "brick_constraint")

# Solve the problem
status = model.solve()

print(model)
print(status)
print(model.objective.value())

for var in model.variables():
    print(f"{var.name}: {var.value()}")
