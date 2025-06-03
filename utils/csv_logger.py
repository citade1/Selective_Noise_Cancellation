import csv
import os

class CSVLogger:
    def __init__(self, file_path, fieldnames):
        self.file_path = file_path
        self.fieldnames = fieldnames
        self.file = open(file_path, mode="a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)

        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            self.writer.writeheader()
        
    def log(self, row_dict):
        self.writer.writerow(row_dict)
    
    def close(self):
        self.file.close()