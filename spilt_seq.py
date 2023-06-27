"""
@Project : Multi-LSTM
@File    : spilt_seq.py
@Author  : endeavor
"""
import os


def spilt_seq(path, left_path, right_path):
    if not os.path.exists(left_path):
        os.makedirs(left_path)
    if not os.path.exists(right_path):
        os.makedirs(right_path)
    filenames = os.listdir(path)
    print(filenames)
    for filename in filenames:
        dir = path + '/' + filename
        left_data = []
        right_data = []
        with open(dir, 'r') as f:
            # left_data.append(f.read(15))
            left_data.append(f.read(29))
            f.close()
        with open(dir, 'r') as f1:
            # right_data.append(f1.read()[16:31])
            right_data.append(f1.read()[32:61])
            f1.close()
            with open(left_path + '/' + filename, 'w') as file:
                str1 = str(left_data)
                str1 = str1.replace('[', '')
                str1 = str1.replace(']', '')
                str1 = str1.replace('\'', '')
                str1 = str1.replace('\'', '')
                # str1 = str1.rstrip()
                str1 = str1[::-1]
                file.write(str1)
                file.close()
            with open(right_path + '/' + filename, 'w') as file:
                str1 = str(right_data)
                str1 = str1.replace('[', '')
                str1 = str1.replace(']', '')
                str1 = str1.replace('\'', '')
                str1 = str1.replace('\'', '')
                # str1 = str1.rstrip()
                # str1 = str1[::-1]
                file.write(str1)
                file.close()

def spilt(path, path_new): # 倒序全序列
    if not os.path.exists(path_new):
        os.makedirs(path_new)
    filenames = os.listdir(path)
    print(filenames)
    for filename in filenames:
        dir = path + '/' + filename
        all_data = []
        with open(dir, 'r') as f1:
            # right_data.append(f1.read()[16:31])
            all_data.append(f1.read())
            f1.close()
            with open(path_new + '/' + filename, 'w') as file:
                str1 = str(all_data)
                str1 = str1.replace('[', '')
                str1 = str1.replace(']', '')
                str1 = str1.replace('\'', '')
                str1 = str1.replace('\'', '')
                # str1 = str1.rstrip()
                str1 = str1[::-1]
                file.write(str1)
                file.close()


if __name__ == '__main__':
    spilt('RLL/neg', 'neg')
    spilt('RLL/pos', 'pos')
