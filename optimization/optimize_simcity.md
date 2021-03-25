# Applying Linear Optimization Sim City Build It

Optimization is a powerful tool for addressing a wide variety of problems including
financial portfolio selection, marketing budget allocation, and inventory and logistics.

Linear programming, or linear optimization, is optimization where the objective function and constraints are linear in terms of decision variables.

In this project, we apply linear optimization to sample game problem that has relevant business context: optimizing
sourcing of raw materials and production of finished goods to maximize profit.

Sim City Build It is a mobile game where players can create a city of their design
by creating the materials needed to develop and pay for building upgrades. In the game,
there are a finite set of materials that can be created, each with it's own production time.
Materials can be combined to create more elaborate items.

The game represents a set of constraints that keeps users engaged in the app,
spending on boosts that can speed up production by limiting the amount of items and resources
that can be produced at any one time.

This led to to wonder what the optimal selection of resources and items were worth producing
in order to maximize profit. The variables were specific to the game, but the constraints of time and money
are applicable to many logistics and real-world problems. Using linear optimization,
I attempted to figure out the optimal selection of items in order to maximize profit.

Optimization consists of an objective function that is either minimized or maximized,
and a set of constraints. Linear objective functions must follow the form:

c1x1 + c2x2 + ··· + cnxn
