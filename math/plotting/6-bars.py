#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """plots a stacked bar graph of fruit per person"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(people))

    plt.bar(x, fruit[0], width=0.5, color='red', label='apples')
    plt.bar(x, fruit[1], width=0.5, color='yellow', label='bananas',
            bottom=fruit[0])
    plt.bar(x, fruit[2], width=0.5, color='#ff8000', label='oranges',
            bottom=fruit[0] + fruit[1])
    plt.bar(x, fruit[3], width=0.5, color='#ffe5b4', label='peaches',
            bottom=fruit[0] + fruit[1] + fruit[2])

    plt.xticks(x, people)
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(range(0, 90, 10))
    plt.legend()

    plt.show()
