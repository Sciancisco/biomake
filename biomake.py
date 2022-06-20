
import numpy as np
import yeadon


O = np.zeros(3)
_1_ = np.identity(3)

def parallel_axis_theorem(I0: np.array[np.float, np.float], R: np.array[np.float], m: float):
    return I0 + m * (R @ R * _1_ - R @ R.T)  # danke Wikipedia [https://en.wikipedia.org/wiki/Parallel_axis_theorem]


class BioModSegment:

    def __init__(
        self,
        label: str,
        parent: str,
        rt: np.array,
        xyz: np.array,
        translations: str,
        rotations: str,
        com: np.array,
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
            mod += f"\trt {BioModSegment.format_vec(self.rt)} xyz {BioModSegment.format_vec(self.xyz)}\n"
            if self.translations:
                self.translations += f"\ttranslations {self.translations}\n"
            if self.rotations:
                self.rotations += f"\trotations {self.rotations}\n"
            mod += f"\tcom {BioModSegment.format_vec(self.com)}\n"
            mod += f"\tmass {self.mass}\n"
            mod += f"\tinertia\n" + BioModSegment.format_mat(self.inertia, leading="\t\t") + "\n"
            mod += "endsegment\n"

            return mod

    @staticmethod
    def format_vec(vec):
        return f"{vec[0]} {vec[1]} {vec[2]}"

    @staticmethod
    def format_mat(mat, leading=""):
        return f"{leading}{mat[0]} {mat[1]} {mat[2]}\n" \
               f"{leading}{mat[3]} {mat[4]} {mat[5]}\n" \
               f"{leading}{mat[6]} {mat[7]} {mat[8]}"


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
    def get_origin(human: yeadon.Human) -> np.array:
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

        segment = yeadon.segment.Segment(
            '',
            O,
            _1_,
            human.T.solids + human.C.solids[:2],  # nipple, shoulder, acromion
            O
        )
        com = segment.rel_center_of_mass.reshape(3)
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

    @staticmethod
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the Thorax in the global frame centered at Pelvis' COM."""
        return human.T.pos.reshape(3) - human.P.center_of_mass.reshape(3)


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

        segment = yeadon.segment.Segment(
            '', O,
            _1_,
            human.C.solids[2:],  # beneath nose, top of ear, top of head
            O
        )
        com = segment.rel_center_of_mass.reshape(3)
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

    @staticmethod
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the Head in the global frame centered at Pelvis' COM."""
        return human.C.solids[2].pos.reshape(3) - human.P.center_of_mass.reshape(3)


class LeftShoulder(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'LeftShoulder'
        parent = 'Thorax'
        rt = O
        xyz = LeftShoulder.get_origin(human) - Thorax.get_origin(human)
        translations = ''
        com = human.A1.solids[0].rel_center_of_mass.reshape(3)
        mass = human.A1.solids[0].mass
        inertia = human.A1.solids[0].rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the LeftShoulder in the global frame centered at Pelvis' COM."""
        return human.A1.solids[0].pos.reshape(3) - human.P.center_of_mass.reshape(3)


class LeftUpperArm(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'LeftUpperArm'
        parent = 'LeftShoulder'
        rt = O
        xyz = LeftUpperArm.get_origin(human) - Thorax.get_origin(human)
        translations = ''
        com = human.A1.solids[1].rel_center_of_mass.reshape(3)
        mass = human.A1.solids[1].mass
        inertia = human.A1.solids[1].rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the LeftUpperArm in the global frame centered at Pelvis' COM."""
        return human.A1.solids[1].pos.reshape(3) - human.P.center_of_mass.reshape(3)


class LeftForearmAndHand(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'LeftForeArmAndHand'
        parent = 'LeftUpperArm'
        rt = O
        xyz = LeftForearmAndHand.get_origin(human) - LeftUpperArm.get_origin(human)
        translations = ''
        com = human.A2.rel_center_of_mass.reshape(3)
        mass = human.A2.mass
        inertia = human.A2.rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the LeftForearmAndHand in the global frame centered at Pelvis' COM."""
        return human.A2.pos.reshape(3) - human.P.center_of_mass.reshape(3)


class RightShoulder(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'RightShoulder'
        parent = 'Thorax'
        rt = O
        xyz = RightShoulder.get_origin(human) - Thorax.get_origin(human)
        translations = ''
        com = human.B1.solids[0].rel_center_of_mass.reshape(3)
        mass = human.B1.solids[0].mass
        inertia = human.B1.solids[0].rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the RightShoulder in the global frame centered at Pelvis' COM."""
        return human.B1.solids[0].pos.reshape(3) - human.P.center_of_mass.reshape(3)


class RightUpperArm(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'RightUpperArm'
        parent = 'RightShoulder'
        rt = O
        xyz = RightUpperArm.get_origin(human) - Thorax.get_origin(human)
        translations = ''
        com = human.B1.solids[1].rel_center_of_mass.reshape(3)
        mass = human.B1.solids[1].mass
        inertia = human.B1.solids[1].rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the RightUpperArm in the global frame centered at Pelvis' COM."""
        return human.B1.solids[1].pos.reshape(3) - human.P.center_of_mass.reshape(3)


class RightForearmAndHand(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'zy'
    ):
        label = 'RightForeArmAndHand'
        parent = 'RightUpperArm'
        rt = O
        xyz = RightForearmAndHand.get_origin(human) - RightUpperArm.get_origin(human)
        translations = ''
        com = human.B2.rel_center_of_mass.reshape(3)
        mass = human.B2.mass
        inertia = human.B2.rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the RightForearmAndHand in the global frame centered at Pelvis' COM."""
        return human.B2.pos.reshape(3) - human.P.center_of_mass.reshape(3)


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
        com = human.J1.rel_center_of_mass.repshape(3)
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the LeftTigh in the global frame centered at Pelvis' COM."""
        return human.J1.pos.reshape(3) - human.P.center_of_mass.reshape(3)


class LeftShankAndFoot(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'LeftShankAndFoot'
        parent = 'LeftTigh'
        rt = O
        xyz = LeftShankAndFoot.get_origin(human) - LeftThigh.get_origin(human)
        translations = ''
        com = human.J2.rel_center_of_mass.reshape(3)
        mass = human.J2.mass
        inertia = human.J2.rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the LeftShankAndFoot in the global frame centered at Pelvis' COM."""
        return human.J2.pos.reshape(3) - human.P.center_of_mass.reshape(3)


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
        com = human.K1.rel_center_of_mass.repshape(3)
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the RightTigh in the global frame centered at Pelvis' COM."""
        return human.K1.pos.reshape(3) - human.P.center_of_mass.reshape(3)


class RightShankAndFoot(BioModSegment):

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = ''
    ):
        label = 'RightShankAndFoot'
        parent = 'RightTigh'
        rt = O
        xyz = RightShankAndFoot.get_origin(human) - RightThigh.get_origin(human)
        translations = ''
        com = human.K2.rel_center_of_mass.reshape(3)
        mass = human.K2.mass
        inertia = human.K2.rel_inertia
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
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the RightShankAndFoot in the global frame centered at Pelvis' COM."""
        return human.K2.pos.reshape(3) - human.P.center_of_mass.reshape(3)


class Tighs(BioModSegment):
    """The tighs of a human if they must remain together."""

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy',
        is_symetric: bool = False
    ):
        label = 'Tighs'
        parent = 'Pelvis'
        rt = O
        xyz = Tighs.get_origin(human) - Pelvis.get_origin(human)
        translations = ''

        mass = human.J1.mass + human.K1.mass

        com_global = (
                (human.J1.mass * human.J1.center_of_mass + human.K1.mass * human.K1.center_of_mass) / mass
        ).reshape(3)
        if is_symetric:  # then center the COM of the tighs
            com_global[:2] = 0.
        com = com_global - xyz

        inertia = parallel_axis_theorem(0, human.J1.center_of_mass-com_global, human.J1.mass) \
                    + parallel_axis_theorem(0, human.K1.center_of_mass-com_global, human.K1.mass)

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
        self.is_symetric = is_symetric

    @staticmethod
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the Tighs in the global frame centered at Pelvis' COM."""
        return human.P.pos.reshape(3) - human.P.center_of_mass.reshape(3)


class ShanksAndFeet(BioModSegment):
    """The shanks and feet of a human if they must remain together."""

    def __init__(
        self,
        human: yeadon.Human,
        rotations: str = 'xy',
        is_symetric: bool = False
    ):
        label = 'ShankAndFeet'
        parent = 'Tighs'
        rt = O
        xyz = LeftShankAndFoot.get_origin(human) - Tighs.get_origin(human)
        translations = ''

        mass = human.J2.mass + human.K2.mass

        com_global = (
                (human.J2.mass * human.J2.center_of_mass + human.K2.mass * human.K2.center_of_mass) / mass
        ).reshape(3)
        if is_symetric:  # then center the COM of the tighs
            com_global[:2] = 0.
        com = com_global - xyz

        inertia = parallel_axis_theorem(0, human.J2.center_of_mass-com_global, human.J2.mass) \
                    + parallel_axis_theorem(0, human.K2.center_of_mass-com_global, human.K2.mass)

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
        self.is_symetric = is_symetric

    @staticmethod
    def get_origin(human: yeadon.Human) -> np.array:
        """Get the origin of the ShanksAndFeet in the global frame centered at Pelvis' COM."""
        return (human.J1.pos + human.K1.pos).reshape(3) / 2. - human.P.center_of_mass.reshape(3)


class BioModHuman:

    def __init__(self, human: yeadon.Human):
        self.head = Head(human)
        self.thorax = Thorax(human)
        self.pelvis = Pelvis(human)
        self.right_shoulder = RightShoulder(human)
        self.right_upper_arm = RightUpperArm(human)
        self.right_forearm_hand = RightForearmAndHand
        self.left_shoulder = LeftShoulder(human)
        self.left_upper_arm = LeftUpperArm(human)
        self.left_forearm_hand = LeftForearmAndHand(human)
        self.right_thigh = RightThigh(human)
        self.right_shank_foot = RightShankAndFoot(human)
        self.left_thigh = LeftThigh(human)
        self.left_shank_foot = LeftShankAndFoot(human)

    def __str__(self):
        return "TODO"
