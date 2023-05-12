import os
import sys
import itertools
import numpy as np
from minizinc import Instance, Model, Solver
import enum
import pandas as pd

E = enum.Enum("E", ["Greater", "Less"])
A = enum.Enum("A", ["Inc", "Dec"])


class TruthTableGetter:

    def __init__(self, model_path='./minizinc'):
        model_path = os.path.join(model_path, "monotone-function.mzn")
        print('loading %s' % model_path)
        self.functions_generator = Model(model_path)
        self.optimizer = Solver.lookup("coinbc")

    def calc(self, inequalities_, composition_):
        instance = Instance(self.optimizer, self.functions_generator)
        instance['functions'] = inequalities
        instance['composition'] = composition_
        result = instance.solve()
        print('found %d solutions' % len(result))
        return result['x'], result['y']


if __name__ == '__main__':

    composition = [A.Inc, A.Dec]

    print(composition)
    table_getter = TruthTableGetter()

    combinations = itertools.product([E.Greater, E.Less], repeat=2)
    truth_table = dict()
    for inequalities in combinations:
        print(inequalities)
        entry = table_getter.calc(inequalities, composition)
        print(entry)
        key = tuple(map(lambda x: 1 if x == E.Greater else 0, list(inequalities)))
        truth_table[key] = np.int0(list(itertools.chain(*entry)))

    # print(truth_table)
    df = pd.DataFrame.from_dict(truth_table, orient='index')
    print(df)
    print(np.array(df.sum(axis=0)))
    df.to_excel('truth_table.xlsx')
