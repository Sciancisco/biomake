from typing import Annotated, Literal, TypeVar
import numpy.typing as npt

import numpy as np
import yeadon


# From [https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype]
DType = TypeVar("DType", bound=np.generic)
Vec2 = Annotated[npt.NDArray[DType], Literal[2]]
Vec3 = Annotated[npt.NDArray[DType], Literal[3]]
Mat3x3 = Annotated[npt.NDArray[DType], Literal[3, 3]]


O = np.zeros(3)


class BioModSegment:

    def __init__(
        self,
        label: str,
        parent: str,
        rt: Vec3,
        xyz: Vec3,
        translations: str,
        rotations: str,
        com: Vec3,
        mass: float,
        inertia: Mat3x3,
        rangesQ: list[Vec2] = None,
        mesh: list[Vec3] = [(0, 0, 0)],
        meshfile: str = None,
        meshcolor: Vec3 = None,
        meshscale: Vec3 = None,
        meshrt: Vec3 = None,
        meshxyz: Vec3 = None,
        patch: list[Vec3] = None
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
        self.rangesQ = rangesQ
        self.mesh = mesh
        self.meshfile = meshfile
        self.meshcolor = meshcolor
        self.meshscale = meshscale
        self.meshrt = meshrt
        self.meshxyz = meshxyz
        self.patch = patch

    def __str__(self):
        mod = f"segment {self.label}\n"
        mod += f"\tparent {self.parent}\n"
        mod += f"\trt {BioModSegment.format_vec(self.rt)} xyz {BioModSegment.format_vec(self.xyz)}\n"
        if self.translations:
            self.translations += f"\ttranslations {self.translations}\n"
        if self.rotations:
            self.rotations += f"\trotations {self.rotations}\n"
        if self.rangesQ:
            mod += f"\trangesQ\n"
            for r in self.rangesQ:
                mod += f"\t\t{BioModSegment.format_vec(r)}\n"
        mod += f"\tcom {BioModSegment.format_vec(self.com)}\n"
        mod += f"\tmass {self.mass}\n"
        mod += f"\tinertia\n" + BioModSegment.format_mat(self.inertia, leading="\t\t") + "\n"
        if self.meshfile:
            mod += f"\tmeshfile {self.meshfile}\n"
        elif self.mesh:
            for m in self.mesh:
                mod += f"\tmesh {BioModSegment.format_vec(m)}\n"
        if self.meshcolor:
            mod += f"\tmeshcolor {BioModSegment.format_vec(self.meshcolor)}\n"
        if self.meshscale:
            mod += f"\tmeshscale {BioModSegment.format_vec(self.meshscale)}\n"
        if self.meshrt and self.meshxyz:
            mod += f"\tmeshrt {BioModSegment.format_vec(self.meshscale)} xyz {BioModSegment.format_vec(self.meshxyz)}\n"
        if self.patch:
            for p in self.patch:
                mod += f"\tpatch {BioModSegment.format_vec(p)}\n"
        mod += "endsegment"

        return mod

    @staticmethod
    def format_vec(vec):
        return ("{} " * len(vec)).format(*vec)[:-1]  # fancy

    @staticmethod
    def format_mat(mat: Mat3x3, leading=""):
        return f"{leading}{mat[0, 0]} {mat[0, 1]} {mat[0, 2]}\n" \
               f"{leading}{mat[1, 0]} {mat[1, 1]} {mat[1, 2]}\n" \
               f"{leading}{mat[2, 0]} {mat[2, 1]} {mat[2, 2]}"


class Pelvis(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        translations: str = 'xyz',
        rotations: str ='xyz'
    ):
        label = 'Pelvis'
        parent = 'ROOT'
        rt = O
        xyz = Pelvis.get_origin(human)
        com = O
        mass = human.P.mass
        inertia = human.P.rel_inertia  # umbilicus, lowest front rib
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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Pelvis in the global frame centered at Pelvis' COM."""
        return O


class Thorax(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'Thorax'
        parent = 'Pelvis'
        rt = O
        xyz = Thorax.get_origin(human) - Pelvis.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('T', 's3', 's4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Thorax.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Thorax in the global frame centered at Pelvis' COM."""
        return np.asarray(human.T.pos - human.P.center_of_mass).reshape(3)


class Head(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'Head'
        parent = 'Thorax'
        rt = O
        xyz = Head.get_origin(human) - Thorax.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('s5', 's6', 's7'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Head.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Head in the global frame centered at Pelvis' COM."""
        length = human.C.solids[0].height + human.C.solids[1].height
        dir = human.C.end_pos - human.C.pos
        dir = dir / np.linalg.norm(dir)
        pos = human.C.pos + length * dir
        return np.asarray(pos - human.P.center_of_mass).reshape(3)


class LeftUpperArm(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'LeftUpperArm'
        parent = 'Thorax'
        rt = O
        xyz = LeftUpperArm.get_origin(human) - Thorax.get_origin(human)
        translations = ''

        com = np.asarray(human.A1.rel_center_of_mass).reshape(3)
        mass = human.A1.mass
        inertia = human.A1.rel_inertia

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftUpperArm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.A1.pos - human.P.center_of_mass).reshape(3)


class LeftForearm(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'LeftForearm'
        parent = 'LeftUpperArm'
        rt = O
        xyz = LeftForearm.get_origin(human) - LeftUpperArm.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('a2', 'a3'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftForearm.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftForearm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.A2.pos - human.P.center_of_mass).reshape(3)


class LeftHand(BioModSegment):

    def __init__(self, human: yeadon.Human, rotations: str = ''):
        label = 'LeftHand'
        parent = 'LeftForearm'
        rt = O
        xyz = LeftHand.get_origin(human) - LeftForearm.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('a4', 'a5', 'a6'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftHand.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human):
        """Get the origin of the LeftHand in the global frame centered at Pelvis' COM."""
        length = human.A2.solids[0].height + human.A2.solids[1].height
        dir = human.A2.end_pos - human.A2.pos
        dir = dir / np.linalg.norm(dir)
        pos = human.A2.pos + length * dir
        return np.asarray(pos - human.P.center_of_mass).reshape(3)


class RightUpperArm(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'RightUpperArm'
        parent = 'Thorax'
        rt = O
        xyz = RightUpperArm.get_origin(human) - Thorax.get_origin(human)
        translations = ''
        com = np.asarray(human.B1.rel_center_of_mass).reshape(3)
        mass = human.B1.mass
        inertia = human.B1.rel_inertia
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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightUpperArm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.B1.pos - human.P.center_of_mass).reshape(3)


class RightForearm(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'RightForearm'
        parent = 'RightUpperArm'
        rt = O
        xyz = RightForearm.get_origin(human) - RightUpperArm.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('b2', 'b3'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightForearm.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightForearm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.B2.pos - human.P.center_of_mass).reshape(3)


class RightHand(BioModSegment):

    def __init__(self, human: yeadon.Human, rotations: str = ''):
        label = 'RightHand'
        parent = 'RightForearm'
        rt = O
        xyz = RightHand.get_origin(human) - RightForearm.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('b4', 'b5', 'b6'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightHand.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human):
        """Get the origin of the RightHand in the global frame centered at Pelvis' COM."""
        length = human.B2.solids[0].height + human.B2.solids[1].height
        dir = human.B2.end_pos - human.B2.pos
        dir = dir / np.linalg.norm(dir)
        pos = human.B2.pos + length * dir
        return np.asarray(pos - human.P.center_of_mass).reshape(3)


class LeftThigh(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy'
    ):
        label = 'LeftTigh'
        parent = 'Pelvis'
        rt = O
        xyz = LeftThigh.get_origin(human) - Pelvis.get_origin(human)
        translations = ''
        com = np.asarray(human.J1.rel_center_of_mass).reshape(3)
        mass = human.J1.mass
        inertia = human.J1.rel_inertia
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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftTigh in the global frame centered at Pelvis' COM."""
        return np.asarray(human.J1.pos - human.P.center_of_mass).reshape(3)


class LeftShank(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'LeftShank'
        parent = 'LeftTigh'
        rt = O
        xyz = LeftShank.get_origin(human) - LeftThigh.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('j3', 'j4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftShank.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftShank in the global frame centered at Pelvis' COM."""
        return np.asarray(human.J2.pos - human.P.center_of_mass).reshape(3)


class LeftFoot(BioModSegment):

    def __init__(self, human: yeadon.Human, rotations: str = ''):
        label = 'LeftFoot'
        parent = 'LeftShank'
        rt = O
        xyz = LeftFoot.get_origin(human) - LeftShank.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('j5', 'j6', 'j7', 'j8'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftFoot.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human):
        """Get the origin of the LeftFoot in the global frame centered at Pelvis' COM."""
        length = human.J2.solids[0].height + human.J2.solids[1].height
        dir = human.J2.end_pos - human.J2.pos
        dir = dir / np.linalg.norm(dir)
        pos = human.J2.pos + length * dir
        return np.asarray(pos - human.P.center_of_mass).reshape(3)


class RightThigh(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy'
    ):
        label = 'RightTigh'
        parent = 'Pelvis'
        rt = O
        xyz = RightThigh.get_origin(human) - Pelvis.get_origin(human)
        translations = ''
        com = np.asarray(human.K1.rel_center_of_mass).reshape(3)
        mass = human.K1.mass
        inertia = human.K1.rel_inertia
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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightTigh in the global frame centered at Pelvis' COM."""
        return np.asarray(human.K1.pos - human.P.center_of_mass).reshape(3)


class RightShank(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'RightShank'
        parent = 'RightTigh'
        rt = O
        xyz = RightShank.get_origin(human) - RightThigh.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('k3', 'k4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightShank.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightShank in the global frame centered at Pelvis' COM."""
        return np.asarray(human.K2.pos - human.P.center_of_mass).reshape(3)


class RightFoot(BioModSegment):

    def __init__(self, human: yeadon.Human, rotations: str = ''):
        label = 'RightFoot'
        parent = 'RightShank'
        rt = O
        xyz = RightFoot.get_origin(human) - RightShank.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('k5', 'k6', 'k7', 'k8'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightFoot.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human):
        """Get the origin of the RightFoot in the global frame centered at Pelvis' COM."""
        length = human.K2.solids[0].height + human.K2.solids[1].height
        dir = human.K2.end_pos - human.K2.pos
        dir = dir / np.linalg.norm(dir)
        pos = human.K2.pos + length * dir
        return np.asarray(pos - human.P.center_of_mass).reshape(3)


class Thighs(BioModSegment):
    """The tighs of a human if they must remain together."""

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy',
    ):
        label = 'Thighs'
        parent = 'Pelvis'
        rt = O
        xyz = Thighs.get_origin(human) - Pelvis.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('J1', 'K1'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Thighs.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Tighs in the global frame centered at Pelvis' COM."""
        return np.asarray(human.P.pos - human.P.center_of_mass).reshape(3)


class Shanks(BioModSegment):
    """The shanks and feet of a human if they must remain together."""

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy',
    ):
        label = 'Shanks'
        parent = 'Thighs'
        rt = O
        xyz = Shanks.get_origin(human) - Thighs.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('j3', 'j4', 'k3', 'k4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Shanks.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the ShanksAndFeet in the global frame centered at Pelvis' COM."""
        return np.asarray((human.J2.pos + human.K2.pos) / 2. - human.P.center_of_mass).reshape(3)


class Feet(BioModSegment):
    """The shanks and feet of a human if they must remain together."""

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy',
    ):
        label = 'Feet'
        parent = 'Shanks'
        rt = O
        xyz = Feet.get_origin(human) - Shanks.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('j5', 'j6', 'j7', 'j8', 'k5', 'k6', 'k7', 'k8'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Feet.get_origin(human)

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

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Feet in the global frame centered at Pelvis' COM."""
        length = ( human.J2.solids[0].height + human.J2.solids[1].height
                  + human.K2.solids[0].height + human.K2.solids[1].height ) / 2.
        dir_J = human.K2.end_pos - human.K2.pos
        dir_K = human.K2.end_pos - human.K2.pos
        dir = (dir_J + dir_K) / 2.
        dir = dir / np.linalg.norm(dir)
        pos = (human.J2.pos + human.K2.pos) / 2. + length * dir
        return np.asarray(pos - human.P.center_of_mass).reshape(3)


class BioModHuman:

    def __init__(self, human: yeadon.Human):
        self.head = Head(human)
        self.thorax = Thorax(human)
        self.pelvis = Pelvis(human)
        self.right_upper_arm = RightUpperArm(human)
        self.right_forearm = RightForearm(human)
        self.right_hand = RightHand(human)
        self.left_upper_arm = LeftUpperArm(human)
        self.left_forearm = LeftForearm(human)
        self.left_hand = LeftHand(human)
        self.right_thigh = RightThigh(human)
        self.right_shank = RightShank(human)
        self.right_foot = RightFoot(human)
        self.left_thigh = LeftThigh(human)
        self.left_shank = LeftShank(human)
        self.left_foot = LeftFoot(human)

    def __str__(self):
        biomod = "version 4\n\nroot_actuated 0\nexternal_forces 0\n\n"
        biomod += f"{self.pelvis}\n\n"
        biomod += f"{self.thorax}\n\n"
        biomod += f"{self.head}\n\n"
        biomod += f"{self.right_upper_arm}\n\n"
        biomod += f"{self.right_forearm}\n\n"
        biomod += f"{self.right_hand}\n\n"
        biomod += f"{self.left_upper_arm}\n\n"
        biomod += f"{self.left_forearm}\n\n"
        biomod += f"{self.left_hand}\n\n"
        biomod += f"{self.right_thigh}\n\n"
        biomod += f"{self.right_shank}\n\n"
        biomod += f"{self.right_foot}\n\n"
        biomod += f"{self.left_thigh}\n\n"
        biomod += f"{self.left_shank}\n\n"
        biomod += f"{self.left_foot}\n"

        return biomod


class BioModHumanFusedLegs:

    def __init__(self, human: yeadon.Human):
        self.head = Head(human)
        self.thorax = Thorax(human)
        self.pelvis = Pelvis(human)
        self.right_upper_arm = RightUpperArm(human)
        self.right_forearm = RightForearm(human)
        self.right_hand = RightHand(human)
        self.left_upper_arm = LeftUpperArm(human)
        self.left_forearm = LeftForearm(human)
        self.left_hand = LeftHand(human)
        self.thighs = Thighs(human)
        self.shanks = Shanks(human)
        self.feet = Feet(human)


    def __str__(self):
        biomod = "version 4\n\nroot_actuated 0\nexternal_forces 0\n\n"
        biomod += f"{self.pelvis}\n\n"
        biomod += f"{self.thorax}\n\n"
        biomod += f"{self.head}\n\n"
        biomod += f"{self.right_upper_arm}\n\n"
        biomod += f"{self.right_forearm}\n\n"
        biomod += f"{self.right_hand}\n\n"
        biomod += f"{self.left_upper_arm}\n\n"
        biomod += f"{self.left_forearm}\n\n"
        biomod += f"{self.left_hand}\n\n"
        biomod += f"{self.thighs}\n\n"
        biomod += f"{self.shanks}\n\n"
        biomod += f"{self.feet}\n\n"

        return biomod


if __name__ == '__main__':
    import sys
    measurements = sys.argv[1]
    human = yeadon.Human(measurements)
    biohuman = BioModHumanFusedLegs(human)
    print(biohuman)
