from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    def __init__(self, label):
        self.label = label
        self.items = []


class Tikzlayer(Layer):
    def __init__(self, label):
        super().__init__(label)

    def generate_tikz(self):
        tikz_script = f"\n% Layer {self.label}\n"
        tikz_script += f"\\begin{{pgfonlayer}}{{{self.label}}}\n"
        for item in self.items:
            tikz_script += item.to_tikz()
        tikz_script += f"\\end{{pgfonlayer}}{{{self.label}}}\n"
        return tikz_script
