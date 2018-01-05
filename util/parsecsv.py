"""
Parse CSV file
"""

csv_file_path = '/home/ashish/Code/Forensics/Manhattan/unifi.csv'
import csv

with open(csv_file_path, 'r') as csvfile:
    _reader = csv.reader(csvfile, delimiter=',')
    for row in _reader:
        print(row[0])
        print(int(not(int(row[1]))))
