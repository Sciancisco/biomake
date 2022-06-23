from _xxsubinterpreters import destroy
from typing import Annotated, Literal, TypeVar
import numpy.typing as npt

import numpy as np
import yaml
import yeadon


# From [https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype]
DType = TypeVar("DType", bound=np.generic)
Vec2 = Annotated[npt.NDArray[DType], Literal[2]]
Vec3 = Annotated[npt.NDArray[DType], Literal[3]]
Mat3x3 = Annotated[npt.NDArray[DType], Literal[3, 3]]


O = np.zeros(3)


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    source [https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space]
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def format_vec(vec):
    return ("{} " * len(vec)).format(*vec)[:-1]  # fancy


def format_mat(mat: Mat3x3, leading=""):
    return f"{leading}{mat[0, 0]} {mat[0, 1]} {mat[0, 2]}\n" \
           f"{leading}{mat[1, 0]} {mat[1, 1]} {mat[1, 2]}\n" \
           f"{leading}{mat[2, 0]} {mat[2, 1]} {mat[2, 2]}"


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
            rangesQ: list[Vec2],
            mesh: list[Vec3],
            meshfile: str,
            meshcolor: Vec3,
            meshscale: Vec3,
            meshrt: Vec3,
            meshxyz: Vec3,
            patch: list[Vec3]
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
        mod += f"\trt {format_vec(self.rt)} xyz {format_vec(self.xyz)}\n"
        if self.translations:
            mod += f"\ttranslations {self.translations}\n"
        if self.rotations:
            mod += f"\trotations {self.rotations}\n"
        if self.rangesQ:
            mod += f"\trangesQ\n"
            for r in self.rangesQ:
                mod += f"\t\t{format_vec(r)}\n"
        mod += f"\tcom {format_vec(self.com)}\n"
        mod += f"\tmass {self.mass}\n"
        mod += f"\tinertia\n" + format_mat(self.inertia, leading="\t\t") + "\n"
        if self.meshfile:
            mod += f"\tmeshfile {self.meshfile}\n"
        elif self.mesh:
            for m in self.mesh:
                mod += f"\tmesh {format_vec(m)}\n"
        if self.meshcolor:
            mod += f"\tmeshcolor {format_vec(self.meshcolor)}\n"
        if self.meshscale:
            mod += f"\tmeshscale {format_vec(self.meshscale)}\n"
        if self.meshrt and self.meshxyz:
            mod += f"\tmeshrt {format_vec(self.meshrt)} xyz {format_vec(self.meshxyz)}\n"
        if self.patch:
            for p in self.patch:
                mod += f"\tpatch {format_vec(p)}\n"
        mod += "endsegment"

        return mod


class Pelvis(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            translations: str = '',
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = Pelvis.__name__
        parent = 'ROOT'
        xyz = Pelvis.get_origin(human)
        com = O
        mass = human.P.mass
        inertia = human.P.rel_inertia
        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Pelvis in the global frame centered at Pelvis' COM."""
        return O


class Thorax(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = Thorax.__name__
        parent = Pelvis.__name__
        xyz = Thorax.get_origin(human) - Pelvis.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('T', 's3', 's4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Thorax.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # I can do this because the systems of coordinates are aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the Thorax in the global frame centered at Pelvis' COM."""
        return np.asarray(human.T.pos - human.P.center_of_mass).reshape(3)


class Head(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = Head.__name__
        parent = Thorax.__name__
        xyz = Head.get_origin(human) - Thorax.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('s5', 's6', 's7'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Head.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # I can do this because the systems of coordinates are aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = LeftUpperArm.__name__
        parent = Thorax.__name__
        xyz = LeftUpperArm.get_origin(human) - Thorax.get_origin(human)
        translations = ''

        com = np.asarray(human.A1.rel_center_of_mass).reshape(3)
        mass = human.A1.mass
        inertia = human.A1.rel_inertia

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftUpperArm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.A1.pos - human.P.center_of_mass).reshape(3)


class LeftForearm(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = LeftForearm.__name__
        parent = LeftUpperArm.__name__
        xyz = LeftForearm.get_origin(human) - LeftUpperArm.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('a2', 'a3'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftForearm.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftForearm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.A2.pos - human.P.center_of_mass).reshape(3)


class LeftHand(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = LeftHand.__name__
        parent = LeftForearm.__name__
        xyz = LeftHand.get_origin(human) - LeftForearm.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('a4', 'a5', 'a6'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftHand.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = RightUpperArm.__name__
        parent = Thorax.__name__
        xyz = RightUpperArm.get_origin(human) - Thorax.get_origin(human)
        translations = ''
        com = np.asarray(human.B1.rel_center_of_mass).reshape(3)
        mass = human.B1.mass
        inertia = human.B1.rel_inertia
        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightUpperArm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.B1.pos - human.P.center_of_mass).reshape(3)


class RightForearm(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = RightForearm.__name__
        parent = RightUpperArm.__name__
        xyz = RightForearm.get_origin(human) - RightUpperArm.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('b2', 'b3'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightForearm.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightForearm in the global frame centered at Pelvis' COM."""
        return np.asarray(human.B2.pos - human.P.center_of_mass).reshape(3)


class RightHand(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = RightHand.__name__
        parent = RightForearm.__name__
        xyz = RightHand.get_origin(human) - RightForearm.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('b4', 'b5', 'b6'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightHand.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = LeftThigh.__name__
        parent = Pelvis.__name__
        xyz = LeftThigh.get_origin(human) - Pelvis.get_origin(human)
        translations = ''
        com = np.asarray(human.J1.rel_center_of_mass).reshape(3)
        mass = human.J1.mass
        inertia = human.J1.rel_inertia
        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftTigh in the global frame centered at Pelvis' COM."""
        return np.asarray(human.J1.pos - human.P.center_of_mass).reshape(3)


class LeftShank(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = LeftShank.__name__
        parent = LeftThigh.__name__
        xyz = LeftShank.get_origin(human) - LeftThigh.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('j3', 'j4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftShank.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the LeftShank in the global frame centered at Pelvis' COM."""
        return np.asarray(human.J2.pos - human.P.center_of_mass).reshape(3)


class LeftFoot(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = LeftFoot.__name__
        parent = LeftShank.__name__
        xyz = LeftFoot.get_origin(human) - LeftShank.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('j5', 'j6', 'j7', 'j8'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - LeftFoot.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = RightThigh.__name__
        parent = Pelvis.__name__
        xyz = RightThigh.get_origin(human) - Pelvis.get_origin(human)
        translations = ''
        com = np.asarray(human.K1.rel_center_of_mass).reshape(3)
        mass = human.K1.mass
        inertia = human.K1.rel_inertia
        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightTigh in the global frame centered at Pelvis' COM."""
        return np.asarray(human.K1.pos - human.P.center_of_mass).reshape(3)


class RightShank(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = RightShank.__name__
        parent = RightThigh.__name__
        xyz = RightShank.get_origin(human) - RightThigh.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('k3', 'k4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightShank.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
        )

    @staticmethod
    def get_origin(human: yeadon.Human) -> Vec3:
        """Get the origin of the RightShank in the global frame centered at Pelvis' COM."""
        return np.asarray(human.K2.pos - human.P.center_of_mass).reshape(3)


class RightFoot(BioModSegment):

    def __init__(
            self,
            human: yeadon.Human,
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = RightFoot.__name__
        parent = RightShank.__name__
        xyz = RightFoot.get_origin(human) - RightShank.get_origin(human)
        translations = ''

        mass, com_global, inertia_global = human.combine_inertia(('k5', 'k6', 'k7', 'k8'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - RightFoot.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia_global,  # TODO: I cannot do this because the systems of coordinates aren't aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = Thighs.__name__
        parent = Pelvis.__name__
        xyz = Thighs.get_origin(human) - Pelvis.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('J1', 'K1'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Thighs.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,  # I can do this because the systems of coordinates are aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = Shanks.__name__
        parent = Thighs.__name__
        xyz = Shanks.get_origin(human) - Thighs.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('j3', 'j4', 'k3', 'k4'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Shanks.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,  # I can do this because the systems of coordinates are aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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
            rt: Vec3 = O,
            rotations: str = '',
            rangesQ: list[Vec2] = None,
            mesh: list[Vec3] = [(0, 0, 0)],
            meshfile: str = None,
            meshcolor: Vec3 = None,
            meshscale: Vec3 = None,
            meshrt: Vec3 = None,
            meshxyz: Vec3 = None,
            patch: list[Vec3] = None
    ):
        label = Feet.__name__
        parent = Shanks.__name__
        xyz = Feet.get_origin(human) - Shanks.get_origin(human)
        translations = ''

        mass, com_global, inertia = human.combine_inertia(('j5', 'j6', 'j7', 'j8', 'k5', 'k6', 'k7', 'k8'))
        com = np.asarray(com_global - human.P.center_of_mass).reshape(3) - Feet.get_origin(human)

        BioModSegment.__init__(
            self,
            label=label,
            parent=parent,
            rt=rt,
            xyz=xyz,
            translations=translations,
            rotations=rotations,
            com=com,
            mass=mass,
            inertia=inertia,  # I can do this because the systems of coordinates are aligned.
            rangesQ=rangesQ,
            mesh=mesh,
            meshfile=meshfile,
            meshcolor=meshcolor,
            meshscale=meshscale,
            meshrt=meshrt,
            meshxyz=meshxyz,
            patch=patch
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

    def __init__(self, human: yeadon.Human, gravity: Vec3 = None, **options):
        self.gravity = gravity
        self.head = Head(human, **options[Head.__name__] if Head.__name__ in options else {})
        self.thorax = Thorax(human, **options[Thorax.__name__] if Thorax.__name__ in options else {})
        self.pelvis = Pelvis(human, **options[Pelvis.__name__] if Pelvis.__name__ in options else {})
        self.right_upper_arm = RightUpperArm(human, **options[RightUpperArm.__name__] if RightUpperArm.__name__ in options else {})
        self.right_forearm = RightForearm(human, **options[RightForearm.__name__] if RightForearm.__name__ in options else {})
        self.right_hand = RightHand(human, **options[RightHand.__name__] if RightHand.__name__ in options else {})
        self.left_upper_arm = LeftUpperArm(human, **options[LeftUpperArm.__name__] if LeftUpperArm.__name__ in options else {})
        self.left_forearm = LeftForearm(human, **options[LeftForearm.__name__] if LeftForearm.__name__ in options else {})
        self.left_hand = LeftHand(human, **options[LeftHand.__name__] if LeftHand.__name__ in options else {})
        self.right_thigh = RightThigh(human, **options[RightThigh.__name__] if RightThigh.__name__ in options else {})
        self.right_shank = RightShank(human, **options[RightShank.__name__] if RightShank.__name__ in options else {})
        self.right_foot = RightFoot(human, **options[RightFoot.__name__] if RightFoot.__name__ in options else {})
        self.left_thigh = LeftThigh(human, **options[LeftThigh.__name__] if LeftThigh.__name__ in options else {})
        self.left_shank = LeftShank(human, **options[LeftShank.__name__] if LeftShank.__name__ in options else {})
        self.left_foot = LeftFoot(human, **options[LeftFoot.__name__] if LeftFoot.__name__ in options else {})

    def __str__(self):
        biomod = "version 4\n\nroot_actuated 0\nexternal_forces 0\n\n"
        if self.gravity:
            biomod += f"gravity {format_vec(self.gravity)}\n\n"
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

    def __init__(self, human: yeadon.Human, gravity: Vec3 = None, **options):
        self.gravity = gravity
        self.head = Head(human, **options[Head.__name__] if Head.__name__ in options else {})
        self.thorax = Thorax(human, **options[Thorax.__name__] if Thorax.__name__ in options else {})
        self.pelvis = Pelvis(human, **options[Pelvis.__name__] if Pelvis.__name__ in options else {})
        self.right_upper_arm = RightUpperArm(human, **options[RightUpperArm.__name__] if RightUpperArm.__name__ in options else {})
        self.right_forearm = RightForearm(human, **options[RightForearm.__name__] if RightForearm.__name__ in options else {})
        self.right_hand = RightHand(human, **options[RightHand.__name__] if RightHand.__name__ in options else {})
        self.left_upper_arm = LeftUpperArm(human, **options[LeftUpperArm.__name__] if LeftUpperArm.__name__ in options else {})
        self.left_forearm = LeftForearm(human, **options[LeftForearm.__name__] if LeftForearm.__name__ in options else {})
        self.left_hand = LeftHand(human, **options[LeftHand.__name__] if LeftHand.__name__ in options else {})
        self.thighs = Thighs(human, **options[Thighs.__name__] if Thighs.__name__ in options else {})
        self.shanks = Shanks(human, **options[Shanks.__name__] if Shanks.__name__ in options else {})
        self.feet = Feet(human, **options[Feet.__name__] if Feet.__name__ in options else {})

    def __str__(self):
        biomod = "version 4\n\nroot_actuated 0\nexternal_forces 0\n\n"
        if self.gravity:
            biomod += f"gravity {format_vec(self.gravity)}\n\n"
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


def parse_biomod_options(filename):
    Human = BioModHuman
    human_options = {}
    segments_options = {}

    if not filename:
        return Human, human_options, segments_options

    with open(filename) as f:
        biomod_options = yaml.safe_load(f.read())

    if 'Human' in biomod_options:
        human_options = biomod_options['Human']
        del biomod_options['Human']
        if 'fused' in human_options:
            if human_options['fused']:
                Human = BioModHumanFusedLegs
            del human_options['fused']

    segments_options = biomod_options

    return Human, human_options, segments_options


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Convert yeadon human model to bioMod.")
    parser.add_argument("--meas", required=True, nargs=1, dest="meas", help="measurement file of the human")
    parser.add_argument("--yeadonCFG", nargs=1, dest="yeadonCFG", help="configuration file of the human")
    parser.add_argument("--bioModOptions", nargs=1, dest="bioModOptions", help="option file for the bioMod")
    args = parser.parse_args()

    meas = args.meas[0]
    yeadonCFG = args.yeadonCFG[0] if args.yeadonCFG else None
    bioModOptions = args.bioModOptions[0] if args.bioModOptions else None

    human = yeadon.Human(meas, yeadonCFG)
    BioHuman, human_options, segments_options = parse_biomod_options(bioModOptions)

    biohuman = BioHuman(human, **human_options, **segments_options)

    print(biohuman)
