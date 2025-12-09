import matplotlib.pyplot as plt
import numpy as np
import time

class Realtimeplot:
    def __init__(self, num_robots = 4, object_name = 'velocity', x_lime = 20, y_lime_min = 0.0, y_lime_max = 1.0):
        plt.ion() # Turn on interactive mode
        
        self.num_robots = num_robots
        self.fig, self.ax = plt.subplots() # Create a figure and an axis
        
        self.colors = ['b', 'g', 'r', 'c'] # Colors for the robots
        self.labels = ['robot1', 'robot2', 'robot3', 'robot4']
        self.lines = [self.ax.plot([], [], color=self.colors[i], label=self.labels[i])[0] for i in range(self.num_robots)] # Create lines for the robots
        
        self.ax.set_xlabel('time') # Set x-axis label
        self.ax.set_ylabel(object_name) # Set y-axis label
        self.ax.legend() # Show legend
        
        self.ax.set_xlim(0, x_lime) # Set x-axis limits
        self.ax.set_ylim(y_lime_min, y_lime_max) # Set y-axis limits
        
        self.xdata = [] # Create an empty list for x-axis data
        self.ydata_all = [[] for i in range(self.num_robots)] # Create an empty list for y-axis data
    
    def show(self):
        plt.ioff() # Turn off interactive mode
        plt.show() # Show the plot
        
        
    def update(self,xdata, ydata):
        self.xdata.append(xdata) # Append x-axis data
        for i in range(self.num_robots):
            self.ydata_all[i].append(ydata[i]) # Append y-axis data
            self.lines[i].set_data(self.xdata, self.ydata_all[i]) # Update the data
            
        self.fig.canvas.flush_events() # Update the plot
        self.ax.relim() # Recalculate limits
        