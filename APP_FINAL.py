import tkinter as tk
from tkinter import ttk as ttk
import pandas as pd
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import CLASSIFIER_FINAL
import numpy as np
import math

class Application:

    def __init__(self, master):
        self.master = master
        self.master.title("Walking/Jumping Classifier")

        # Create the file selection button
        self.select_button = ttk.Button(master, text="Select CSV File", command=self.select_file)
        #self.select_button.pack(side=tk.TOP, padx=10, pady=10)
        self.select_button.place(x=450, y=5)

        # Create the graph button
        self.graph_button = ttk.Button(master, text="Generate Output", command=self.graph_data, state=tk.DISABLED)
        #self.graph_button.pack(side=tk.TOP, padx=10, pady=10)
        self.graph_button.place(x=380, y=52)

        self.graph_next_button = ttk.Button(master, text="Generate Next Output", command=self.graph_next_data, state=tk.DISABLED)
        #self.graph_next_button.pack(side=tk.TOP, padx=50, pady=10)
        self.graph_next_button.place(x=520, y=52)

        # Create a frame for the buttons
        self.button_frame = tk.Frame(master)
        #self.button_frame.pack(side=tk.TOP)
        self.button_frame.place(x=500, y=100)

        # Create the figure and canvas for the graph
        self.fig = plt.figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        #self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().place(x=0, y=100, relwidth=1, relheight = 0.9)

        # Initialize the dataframe variable
        self.dataframe = None

        # Apply the custom button style
        self.apply_button_style()

        # initialize the model
        self.model, self.scaler = CLASSIFIER_FINAL.train()
        self.csv_file = "Output.csv"
        info = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                "avgAccel", "maxAccel", "avgTimeAboveAvg", "label (0 = walking, 1 = jumping)"]
        df = pd.DataFrame([info])
        df.to_csv(self.csv_file, mode='a', header=False, index=False)

        self.count = 0

    def apply_button_style(self):
        # Create a style for the buttons
        style = ttk.Style()
        style.configure("GraphApp.TButton", padding=10, font=("Arial", 12))

        # Apply the style to the buttons
        self.select_button.config(style="GraphApp.TButton")
        self.graph_button.config(style="GraphApp.TButton")
        self.graph_next_button.config(style="GraphApp.TButton")

    def select_file(self):
        # Open the file dialog and get the file path
        filepath = askopenfilename(filetypes=[("CSV Files", "*.csv")])

        # Read the CSV into a Pandas dataframe
        self.dataframe = pd.read_csv(filepath)
        #print(self.dataframe)
        self.array = np.array(self.dataframe)
        #print(self.array)
        self.segment = []
        self.segments = []
        for i in range (0, math.floor(len(self.array)/500)):
            for j in range (i*500, (i+1)*500):
                self.segment.append(self.array[j])
            self.segments.append(self.segment)
            self.segment = []
        self.max_count = len(self.segments)




        # Enable the graph button
        self.graph_button.config(state=tk.NORMAL)
        self.graph_next_button.config(state=tk.DISABLED)

    def graph_data(self):
        self.count = 0
        #self.dataframe.set_index(self.dataframe.columns[0], inplace=True)
        if (self.max_count > 0):
            self.graph_next_button.config(state=tk.NORMAL)



        # Clear the figure and plot the data
        self.fig.clear()
        # remove first column
        #for i in range (0,501):
         #   values = self.dataframe.loc[i,:].values
         #   self.segment = pd.concat([self.segment, pd.DataFrame([0,0,0,0,0])])
         #   self.segment.loc[i,0:5] = values
        # remove first row
        #self.segment = self.segment.iloc[1:]
        #print(self.segment)

        final = CLASSIFIER_FINAL.Classify(self.model, self.scaler, pd.DataFrame(self.segments[self.count]))
        final.to_csv(self.csv_file, mode='a', header=False, index=False)
        final = np.array(final)

        #self.dataframe = self.dataframe.iloc[:, 1:]
        #self.dataframe.plot(ax=self.fig.add_subplot(111))


        # Set the x-axis label to the index column name
        #plt.xlabel("time (1/100s)")
        #plt.ylabel("acceleration (m/s^2)")

        sets = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                "avgAccel", "maxAccel", "avgTimeAboveAvg"]

        graph = [final[0,0], final[0,1], final[0,2], final[0,3], final[0,4], final[0,5], final[0,6], final[0,7], final[0,8], final[0,9]]
        plt.rcParams.update({'font.size': 8})
        plt.title("Prediction: " + ("Walking" if final[0,10] == 0 else "Jumping"))
        plt.bar(sets, graph)

        # Redraw the canvas
        self.canvas.draw()
    def graph_next_data(self):

        #self.dataframe.set_index(self.dataframe.columns[0], inplace=True)


        self.count += 1
        if (self.count < self.max_count - 1):
            self.graph_next_button.config(state=tk.NORMAL)
        else:
            self.graph_next_button.config(state=tk.DISABLED)



        # Clear the figure and plot the data
        self.fig.clear()

        final = CLASSIFIER_FINAL.Classify(self.model, self.scaler, pd.DataFrame(self.segments[self.count]))
        final.to_csv(self.csv_file, mode='a', header=False, index=False)
        final = np.array(final)

        #self.dataframe = self.dataframe.iloc[:, 1:]
        #self.dataframe.plot(ax=self.fig.add_subplot(111))
        sets = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                "avgAccel", "maxAccel", "avgTimeAboveAvg"]

        graph = [final[0,0], final[0,1], final[0,2], final[0,3], final[0,4], final[0,5], final[0,6], final[0,7], final[0,8], final[0,9]]
        plt.rcParams.update({'font.size': 8})
        plt.title("Prediction: " + ("Walking" if final[0,10] == 0 else "Jumping"))
        plt.bar(sets, graph)

        # Redraw the canvas
        self.canvas.draw()

# Create the main application window
root = tk.Tk()
root.geometry("1000x1000")


app = Application(root)

# Run the application loop
root.mainloop()
