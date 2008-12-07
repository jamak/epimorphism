class DataManager:
    TPATH = "aer/t.epi"
    TSEEDPATH = "aer/t_seed.epi"

    def __init__(self, cmdcenter):
        self.cmdcenter = cmdcenter

        # load t
        file = open(self.TPATH)
        self.t = []
        for line in file.readlines():
            if(line == "\n"):
                continue
            data = line.split(':')
            data[0] = data[0].strip()
            data[1] = data[1].strip()[1:-1].split(',')
            self.t.append(data)
        file.close()

        # load t_seed
        file = open(self.TSEEDPATH)
        self.t_seed = []
        for line in file.readlines():
            if(line == "\n"):
                continue
            data = line.split(':')
            data[0] = data[0].strip()
            data[1] = data[1].strip()[1:-1].split(',')
            self.t_seed.append(data)
        file.close()

