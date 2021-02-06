class Node:

    def __init__(self, cfg, id_):
        self.cfg = cfg          # Angles in radians for the 6 revolute joints
        self.id = id_            # Index in the kd-tree data
        self.edges = []         # List of tuples (node_id, dist)
                                # of neighboring nodes found with kd-tree
                                # prefer        list > set because
                                # prefer  iter speed > contains check speed


    def __str__(self):
        s = ''
        s += 'Node at ind: {}\n'.format(self.id)
        s += '\tnum edges: {}\n'.format(len(self.edges))
        s += '\tedges (id, dist):\n'
        for edge in self.edges:
            s += '\t\t({} , {:3.3f})\n'.format(edge[0], edge[1])
        # s += '\tcfg: {}\n'.format(self.cfg)
        return s

    def __repr__(self):
        s = ''
        s += 'Node at ind: {}\n'.format(self.id)
        s += '\tnum edges: {}\n'.format(len(self.edges))
        s += '\tedges (id, dist):\n'
        for edge in self.edges:
            s += '\t\t({} , {:3.3f})\n'.format(edge[0], edge[1])
        # s += '\tcfg: {}\n'.format(self.cfg)
        return s

    def addEdge(self, neighbor_id, neighbor_dist):
        assert neighbor_id != self.id, 'tried connecting illegal loop edge'

        if (neighbor_id not in [tuple_[0] for tuple_ in self.edges]):      # no duplicates
            
            self.edges.append((neighbor_id, neighbor_dist))