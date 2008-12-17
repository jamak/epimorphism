import os
import re

class DataManager(object):

    def __init__(self, state):

        files = [file for file in os.listdir("aer") if re.search("epi$", file)]

        # will crash unexplanedly with invalid file
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
                val.append('')
                data.append(val)
            file.close()


        files = [file for file in os.listdir("aer") if re.search("^[^_][^#]+?cu$", file)]

        # will crash unexplanedly with invalid file
        for file_name in files:
            data_name = file_name.split('.')[0].upper()
            file = open("aer/" + file_name)
            contents = file.read()
            file.close()
            if(re.search("EPIMORPH library file", contents)):
                exec("data = self." + data_name + " = []")
                funcs = re.findall("^__device__.+?^}$", contents, re.M | re.S)
                for func in funcs:
                    func_name = re.search("__device__ .+? (\S+)\(", func).group(1)
                    args = [arg.split(" ")[1] for arg in re.search("\(.+\)", func).group(0)[1:-1].split(", ")]
                    clause = "(" + ", ".join(args) + ")"
                    comments = re.findall("//\s*(.+)", func)
                    if(not re.match("EXCLUDE", comments[0])):
                        val = [func_name + clause, [], comments[0]]
                        if(len(comments) == 2):
                            val[1] = [cmd.strip() for cmd in comments[1].strip().split('#')]
                        data.append(val)




