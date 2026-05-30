#!/usr/bin/env python3
"""Module to plot a histogram of student scores."""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plots a histogram of student grades with bins every 10 units."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Define bins every 10 units ranging from 0 to 100
    bins = np.arange(0, 110, 10)

    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(0, 100)
    plt.show()
