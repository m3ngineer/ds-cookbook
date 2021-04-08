from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

# Create the model
model = LpProblem(name="sim-city", sense=LpMaximize)

materials = {
    'metal': {'material': [], 'time': 60, 'revenue': 10, 'discount': 0},
    'nail': {'material': [('metal', 2)], 'time': 300, 'revenue': 80, 'discount': 0},
    'brick': {'material': [('mineral', 2)], 'time': 1200, 'revenue': 190, 'discount': 0},
    'plank': {'material': [('log', 2)], 'time': 1800, 'revenue': 120, 'discount': 0},
    'leek': {'material': [('seed', 2)], 'time': 1200, 'revenue': 160, 'discount': 0},
    'seed': {'material': [], 'time': 720, 'revenue': 30, 'discount': 0},
    'mineral': {'material': [], 'time': 1800, 'revenue': 40, 'discount': 0},
    'log': {'material': [], 'time': 180, 'revenue': 14, 'discount': 0},
    'plastic': {'material': [], 'time': 540, 'revenue': 25, 'discount': 0},
    'chemical': {'material': [], 'time': 7200, 'revenue': 60, 'discount': 0},
    'textile': {'material': [], 'time': 10800, 'revenue': 90, 'discount': 0},
    'hammer': {'material': [('log', 1), ('metal', 1)], 'time': 840, 'revenue': 90, 'discount': 0},
    'measuring_tape': {'material': [('log', 1), ('plastic', 1)], 'time': 1200, 'revenue': 110, 'discount': 0},
    'shovel': {'material': [('metal', 1), ('plastic', 1), ('log', 1)], 'time': 1800, 'revenue': 150, 'discount': 0},
    'chair': {'material': [('log', 2), ('nail', 1), ('hammer', 1)], 'time': 1200, 'revenue': 300, 'discount': 0},
    'table': {'material': [('plank', 1), ('nail', 2), ('hammer', 1)], 'time': 1800, 'revenue': 500, 'discount': 0},
    'cement': {'material': [('mineral', 2), ('chemical', 1)], 'time': 3000, 'revenue': 440, 'discount': 0},
    'spices': {'material': [], 'time': 14400, 'revenue': 110, 'discount': 0},
    'glue': {'material': [('plastic', 1), ('chemical', 2)], 'time': 3600, 'revenue': 440, 'discount': 0},
    'paint': {'material': [('metal', 2), ('mineral', 1), ('chemical', 1)], 'time': 3600, 'revenue': 320, 'discount': 0},
    'flour': {'material': [('seed', 2), ('textile', 2)], 'time': 2700, 'revenue': 570, 'discount': 0},
    'donut': {'material': [('flour', 1), ('spices', 1)], 'time': 2700, 'revenue': 950, 'discount': 0},
    'spatula': {'material': [('metal', 2), ('plastic', 2), ('log', 2)], 'time': 2700, 'revenue': 250},
    'watermelon': {'material': [('seed', 2), ('sapling', 1)], 'time': 5400, 'revenue': 730, 'discount': 0},
    'sapling': {'material': [('seed', 2), ('shovel', 1)], 'time': 5400, 'revenue': 420, 'discount': 0},
    'grass': {'material': [('seed', 1), ('shovel', 1)], 'time': 1800, 'revenue': 310, 'discount': 0},
    'green_smoothie': {'material': [('leek', 1), ('watermelon', 1)], 'time': 1800, 'revenue': 1150, 'discount': 0},
    'hat': {'material': [('textile', 1), ('measuring_tape', 1)], 'time': 3600, 'revenue': 600, 'discount': 0},
    'flour': {'material': [('textile', 2), ('seed', 2)], 'time': 1800, 'revenue': 570, 'discount': 0},
    'cap': {'material': [('textile', 2), ('measuring_tape', 1)], 'time': 3600, 'revenue': 600, 'discount': 0},
    'ladder': {'material': [('plank', 2), ('metal', 2)], 'time': 3600, 'revenue': 420, 'discount': 0},
    'garden_furniture': {'material': [('plank', 2), ('plastic', 2), ('textile', 2)], 'time': 4500, 'revenue': 820, 'discount': 0},
    'sneaker': {'material': [('plastic', 1), ('glue', 1), ('textile', 2)], 'time': 2700, 'revenue': 980, 'discount': 0},
    'watch': {'material': [('plastic', 1), ('glue', 1), ('textile', 2)], 'time': 2700, 'revenue': 580, 'discount': 0},
    'animal_feed': {'material': [], 'time': 21600, 'revenue': 140, 'discount': 0},
    'cream': {'material': [('animal_feed', 1)], 'time': 4500, 'revenue': 440, 'discount': 0},
    'bread_roll': {'material': [('flour', 2), ('cream', 1)], 'time': 3600, 'revenue': 1840, 'discount': 0},
    'corn': {'material': [('mineral', 1), ('seed', 4)], 'time': 3600, 'revenue': 280, 'discount': 0},
    'cheese': {'material': [('animal_feed', 2)], 'time': 6300, 'revenue': 660, 'discount': 0}, #check revenue
    'ice_cream_sandwich': {'material': [('bread_roll', 1), ('cream', 1)], 'time': 2560, 'revenue': 280, 'discount': 0}, #check revenue
}

# Add the objective function to the model
expr, constraint = None, None
click_penalty = 1.1 # penalty for every click that occurs
hours = 12
time_limit = 3600 * hours

vars = {}
for k,v in materials.items():
    print(k)
    var = LpVariable(name=k, lowBound=0)
    expr += v['revenue'] * var
    constraint += var
    vars[k] = var


    production_time = v['time']
    for material in v['material']:
        ingredient = material[0]
        production_time += materials[ingredient]['time'] * material[1]
    model += (var * production_time <= (time_limit) * click_penalty , "{}_constraint".format(k))
    # * (1-v['discount'])

print(expr)
model += expr

total_factory_slots = 35

# Add the constraints to the model
click_penalty = 1.1 # penalty for every click that occurs
hours = 12
total_time = 3600 * hours
# model += (constraint, "slot_constraint")
model += (constraint <= total_factory_slots, "slot_constraint")

# Add individual factory time limit constraints
model += ( vars['nail'] * materials['nail']['time']
            + vars['plank'] * materials['plank']['time']
            + vars['brick'] * materials['brick']['time']
            + vars['cement'] * materials['cement']['time']
            + vars['glue'] * materials['glue']['time']
            + vars['paint'] * materials['paint']['time'] <= total_time, "building_store_constraint")
model += ( vars['hammer'] * materials['hammer']['time'] + \
            vars['measuring_tape'] * materials['measuring_tape']['time'] +
            vars['shovel'] * materials['shovel']['time'] +
            vars['spatula'] * materials['spatula']['time'] +
            vars['ladder'] * materials['ladder']['time'] <= total_time, "hardware_store_constraint")
model += ( vars['leek'] * materials['leek']['time'] + \
            vars['watermelon'] * materials['watermelon']['time'] <= total_time, "farmer_market_constraint")


# TODO: Add in prerequisite order of items into constraint

# Solve the problem
status = model.solve()

print(model)
print(status)
print(model.objective.value())

for var in model.variables():
    print(f"{var.name}: {var.value()}")
