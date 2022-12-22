import csv
import re


FINAL_PATH = 'C:\\datasets\\amazon_for_us\\test-final.csv'
pre_path = "C:\\datasets\\amazon_for_us\\test.csv"
SAVE_PATH_ALL = 'C:\\datasets\\amazon_for_us\\test-final-all.csv'
SAVE_PATH_ALL_LABEL = 'C:\\datasets\\amazon_for_us\\决赛\\label-final-all.txt'


final_test = []
with open(FINAL_PATH, 'r') as final_data:
    writer = csv.reader(final_data)
    for review, label in writer:
        final_test.append((review, label))

pre_test = []
with open(pre_path, 'r') as final_data:
    writer = csv.reader(final_data)
    for review, label in writer:
        pre_test.append((review, label))


final_test_all = final_test[10000:] + pre_test[1:10001]


with open(SAVE_PATH_ALL, 'w', newline='') as student_file:
    writer = csv.writer(student_file)
    writer.writerow(["review", "label"])
    for i in range(20000):
        review, label = final_test_all[i]
        writer.writerow((review, 0))

mark = 0
with open(SAVE_PATH_ALL_LABEL, 'w', newline='') as f:
    for i in range(20000):
        review, label = final_test_all[i]
        if not re.search(r"\d+", label):
            print("error", i)
        f.write(str(label)+'\r\n')
        mark = mark + 1
print("result length", mark)

i = 0
with open("C:\\datasets\\amazon_for_us\\决赛\\label-final-all.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        i = i + 1

print("example length", i)
