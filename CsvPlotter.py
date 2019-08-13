import csv
import matplotlib.pyplot as plt
import numpy as np


EPS = 8

class CsvPlotter:

    def __init__(self, file_name):
        self.read_file(file_name)
        self.x_axis_name = "round"
        self.y_axis_name = "score"


    def read_file(self, file_name):

        with open(file_name, 'r') as f:
            readers = csv.reader(f)
            self.data = {}

            print(readers)

            is_header = True
            for row in readers:
                if(is_header):
                    headers = row
                    for i in range(len(headers)):
                        self.data[headers[i]] = []
                    is_header = False
                else:
                    for i in range(len(headers)):
                        self.data[headers[i]].append(float(row[i]))

    def plot(self, name, title):
        x_axis_data = self.data[self.x_axis_name]

        plt.figure()
        for data_name in self.data:
            if(data_name == self.x_axis_name):
                continue
            plt.plot(x_axis_data, self.data[data_name], label=data_name)
        plt.xlabel(self.x_axis_name)
        plt.ylabel(self.y_axis_name)
        plt.legend()
        plt.title(title)
        plt.savefig(name)
        plt.show()

    def plot_learning_part(self, rounds, learning_duration, title, name):
        x_axis_data = self.data[self.x_axis_name]

        indices, = np.where(np.array(x_axis_data) > rounds - learning_duration)

        plt.figure()
        for data_name in self.data:
            if(data_name == self.x_axis_name):
                continue
            plt.plot(np.array(x_axis_data)[indices], np.array(self.data[data_name])[indices], label=data_name)
        plt.xlabel(self.x_axis_name)
        plt.ylabel(self.y_axis_name)
        plt.legend()
        plt.title(title)
        plt.savefig(name)
        plt.show()



# plotter = CsvPlotter("csv_game_f=4_e=0.5_dr=0.8_lr=0.001_t3.csv")
# plotter.plot("f_4_test", "Custom Snake game test - \n Custom(epsilon=0.5,dr=0.8,lr=0.001),\n Avoid(epsilon=0.5), Avoid(epsilon=0.2), \nAvoid(epsilon=0.001), Avoid(epsilon=0.02)")
# plotter.plot_learning_part(50000, 5000, "Custom Snake game test - learning part  \n Custom(epsilon=0.5,dr=0.8,lr=0.001), \nAvoid(epsilon=0.5), Avoid(epsilon=0.2), \n Avoid(epsilon=0.001), Avoid(epsilon=0.02)", "f_4_test_learn" )



plotter = CsvPlotter("csv_game_f=2_e.csv")
plotter.plot("csv_c_eps_game", "Custom Snake game - \n choose epsilon")
plotter.plot_learning_part(50000, 5000, "Custom Snake game learning part \n Choose epsilon", "csv_c_eps_learn" )

plotter = CsvPlotter("csv_game_f=2_e=0.5_dr.csv")
plotter.plot("csv_c_eps_dr_game", "Custom Snake game - epsilon = 0.5 \n Choose discount rate")
plotter.plot_learning_part(50000, 5000, "Custom Snake game learning part - epsilon = 0.5 \n Choose discount rate", "csv_c_eps_dr_learn")

plotter = CsvPlotter("csv_game_f=2_e=0.5_dr=0.1_lr.csv")
plotter.plot("csv_c_eps_dr_lr_game", "Custom Snake game - epsilon = 0.5, discount rate = 0.1 \n "
                                     "Choose learning rate")
plotter.plot_learning_part(50000, 5000, "Custom Snake game learning part - epsilon = 0.5, "
                                        "discount rate = 0.1 \n Choose learning rate", "csv_c_eps_dr_lr_learn")


# plotter = CsvPlotter("csv_l_eps.csv")
# plotter.plot("csv_l_eps_game", "Linear Snake game - \n choose epsilon")
# plotter.plot_learning_part(5000, 1000, "Linear Snake game learning part \n Choose epsilon", "csv_l_eps_learn" )
#
# plotter = CsvPlotter("csv_l_eps=0.001_dr.csv")
# plotter.plot("csv_l_eps_dr_game", "Linear Snake game - epsilon = 0.001 \n Choose discount rate")
# plotter.plot_learning_part(5000, 1000, "Linear Snake game learning part - epsilon = 0.001 \n Choose discount rate", "csv_l_eps_dr_learn")
#
# plotter = CsvPlotter("csv_l_eps=0.001_dr=0.1_lr.csv")
# plotter.plot("csv_l_eps_dr_lr_game", "Linear Snake game - epsilon = 0.001, discount rate = 0.1 \n Choose learning rate")
# plotter.plot_learning_part(5000, 1000, "Linear Snake game learning part - epsilon = 0.001, discount rate = 0.1 \n Choose learning rate", "csv_l_eps_dr_lr_learn")




