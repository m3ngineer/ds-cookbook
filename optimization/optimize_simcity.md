# Applying Linear Optimization Sim City Build It

Optimization is a powerful tool for addressing a wide variety of problems including
financial portfolio selection, marketing budget allocation, and inventory and logistics.

Linear programming, or linear optimization, is optimization where the objective function and constraints are linear in terms of decision variables.

In this project, we apply linear optimization to maximize the profit for the production of set of raw materials and finished goods. Specifically, I'll be using the set up of a mobile game that I've been playing too
many hours on--Sim City Build It. The

Sim City Build It is a game where players can create a city of their design
by creating the materials needed to develop and pay for building upgrades. In the game,
there are a finite set of materials that can be created, each with its own production time.
Materials can be combined to create more elaborate items. Items can be sold to vendors or
other players in the trade depot for the Sim equivalent of cash, Simoleons.

The game contains a series of constraints which keep users engaged and spending in the app
that include limiting the amount of items and resources that can be produced at any one time,
and adding production times for each resource. While this example represents a simplified problem,
the general objective and constraints are relevant to many business contexts as well.

This led me to wonder what the optimal selection of resources and items were worth producing
in order to maximize profit. Using linear optimization,
I attempted to figure out the optimal selection of items in order to maximize profit.

## Linear Optimization in a Nutshell
Optimization consists of an objective function that is either minimized or maximized,
and a set of constraints. Linear objective functions must follow the form:

c1x1 + c2x2 + ··· + cnxn

Objective functions are subject to a set of linear constaints.

Additionally, the decision variables must be real variables.

## PuLP
There are several open-sourced Python libraries available for linear programming.
[PuLP](https://pypi.org/project/PuLP/) is a high-level library package for linear programming in Python.

In this example, we are trying to attain the highest possible revenue, so this is
a maximization problem.

```
from Pulp import *

model = LpProblem('Sim City Problem', LpMaximize)
```

We can set the variables

```
metal = LpVariable("metal", lowBound = 0)
log = LpVariable("log", lowBound = 0)
seed = LpVariable("seed", lowBound = 0)
```

Add the objective function
```
model += 10 * metal + 14 * log + 30 * seed
```

Add constraints
```
total_factory_slots = 35
model += (metal + log + seed <= total_factory_slots, "slot_constraint")
```

Solve the problem
```
model.solve()
```
