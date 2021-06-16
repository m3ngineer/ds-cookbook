import matplotlib.pyplot as plt

newspaper_cost = 5
customer_price = 7

inventory = 42
demand = 40

profit = demand * customer_price - newspaper_cost * inventory
print(f'${profit}')

def profit_calculator(inventory, demand, cogs=5, price=7):
    profit_per_sale = price - cogs
    if demand < inventory:
        leftover = inventory - demand
        utility = demand * profit_per_sale - leftover * cogs

    if demand >= inventory:
        utility = inventory * profit_per_sale

    return utility

print(f'${profit_calculator(42, 40)}')

x = list(range(100))
y = [profit_calculator(x, 40) for x in x]

plt.plot(x, y)
plt.show()
