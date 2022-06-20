from collections import namedtuple

import numpy as np
import yeadon


Vec = namedtuple('Vec', 'x y z')
Vec.__str__ = lambda self: f"{self.x} {self.y} {self.z}"

O = Vec(0, 0, 0)


class BioModSegment:

    def __init__(
        self,
        label: str,
        parent: str,
        rt: Vec,
        xyz: Vec,
        translations: str,
        rotations: str,
        com: Vec,
        mass: float,
        inertia: np.array
    ):
        self.label = label
        self.parent = parent
        self.rt = rt
        self.xyz = xyz
        self.translations = translations
        self.rotations = rotations
        self.com = com
        self.mass = mass
        self.inertia = inertia

        def __str__(self):
            mod = f"segment {self.label}\n"
            mod += f"\tparent {self.parent}\n"
            mod += f"\trt {self.rt} xyz {self.xyz}\n"
            if self.translations:
                self.translations += f"\ttranslations {self.translations}\n"
            if self.rotations:
                self.rotations += f"\trotations {self.rotations}\n"
            mod += f"\tcom {self.com}\n"
            mod += f"\tmass {self.mass}\n"
            mod += f"\tinertia\n\t\t{self.inertia}\n"
            mod += "endsegment\n"

            return mod


class Pelvis(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        translations: str = 'xyz',
        rotations: str =' xyz'
    ):
        label = 'Pelvis'
        parent = 'ROOT'
        rt = O
        xyz = O
        com = O
        inertia = human.P.rel_inertia
        BioModSegment.__init__(
            self,
            label,
            parent,
            rt,
            xyz,
            translations,
            rotations,
            com,
            mass,
            inertia
        )


class Thorax(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        translations: str = ''
        rotations: str = ''
    ):
        label = 'Thorax'
        parent = 'Pelvis'
        rt = O
        xyz = Vec(*(np.array([0, 0, human.P.length]) - human.P.center_of_mass))

        segment = yeadon.segment.Segment(
            '',
            np.zeros(3),
            np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
            human.T.solids + human.C.solids[:2],
            O
        )
        com = segment.rel_center_of_mass
        mass = segment.mass
        inertia = segment.rel_inertia
        BioModSegment.__init__(
            self,
            label,
            parent,
            rt,
            xyz,
            translations,
            rotations,
            com,
            mass,
            inertia
        )


class Head(BioModSegment):
    pass
