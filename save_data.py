import csv

class csvWriter:

    def __init__(self, field_names, csv_name):
        self.field_names = field_names
        self.rows_array = []
        self.cur_row = {}
        self.file_name = csv_name

    def build_row(self, col_index, score):
        if len(self.cur_row) == len(self.field_names):
            self.rows_array.append(self.cur_row)
            self.cur_row = {}

        self.cur_row[self.field_names[col_index]] = score

    def create_csv(self):
        with open(self.file_name, 'w') as csvfile:
            self.rows_array.append(self.cur_row)

            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            writer.writeheader()

            for row in self.rows_array:
                writer.writerow(row)


