import os
import re


class DataManager(object):
    ''' The DataManager object is resonsible for loading components from
        .epi and .cu library files in the aer directory '''

    def __init__(self, state):

        # load components from files of form *.epi
        files = [file for file in os.listdir("aer") if re.search("^[^_\.][^#]*?.epi$", file)]

        for file_name in files:

            # get component name
            component_name = file_name.split('.')[0]

            # open file & read contents
            file = open("aer/" + file_name)
            contents = file.read()
            file.close()

            # set component
            setattr(self, component_name.upper(), [])
            components = getattr(self, component_name.upper())

            # get all components
            for line in contents.split("\n"):

                # skip blank lines
                if(line == "") : continue

                # parse line
                component = line.split(':')

                # get component
                component[0] = component[0].strip()

                # get defaults
                if(len(component) == 2):
                    component[1] = [cmd.strip() for cmd in component[1].strip().split('#')]
                else:
                    component[1] = []

                # add comment
                component.append(component_name)

                # add component
                components.append(component)

        # load components from files of form *.cu
        files = [file for file in os.listdir("aer") if re.search("^[^_\.][^#]*?cu$", file)]

        for file_name in files:

            # get component name
            component_name = file_name.split('.')[0].upper()
            file = open("aer/" + file_name)
            contents = file.read()
            file.close()

            # only parse library files
            if(re.search("EPIMORPH library file", contents)):

                # set component
                setattr(self, component_name.upper(), [])
                components = getattr(self, component_name.upper())

                # get all function definitions
                funcs = re.findall("^__device__.+?^}$", contents, re.M | re.S)

                for func in funcs:

                    # get function name & args
                    try:
                        func_name = re.search("__device__ .+? (\S+)\(", func).group(1)
                        args = [arg.split(" ")[1] for arg in re.search("\(.+\)", func).group(0)[1:-1].split(", ")]
                    except:
                        print "invalid function definition", func
                        continue

                    # find comments
                    comments = re.findall("//\s*(.+)", func)

                    # default if no comments
                    if(len(comments) == 0) : comments = [""]

                    # skip EXCLUDE funcs
                    if(not re.match("EXCLUDE", comments[0])):

                        # make clause
                        clause = "(" + ", ".join(args) + ")"
                        component = [func_name + clause, [], comments[0]]

                        # get defaults
                        if(len(comments) == 2):
                            component[1] = [cmd.strip() for cmd in comments[1].strip().split('#')]

                        # add component
                        components.append(component)
