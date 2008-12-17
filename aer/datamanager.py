import os
import re

class DataManager(object):

    def __init__(self, state):

        files = [file for file in os.listdir("aer") if re.search("epi$", file)]

        for file_name in files:
            data_name = file_name.split('.')[0]
            file = open("aer/" + file_name)
            exec("data = self." + data_name + " = []")

            for line in file.readlines():
                if(line == "\n"):
                    continue
                val = line.split(':')
                val[0] = val[0].strip()
                val[1] = [cmd.strip() for cmd in val[1].strip().split('#')]
                data.append(val)
            file.close()
