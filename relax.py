#!/usr/bin/env python3
# Tianren 2020
import os, shutil, sys, re
import time, typing
import numpy as np
import scipy.optimize as opt

# customizable variables
run_aims_cmd = '/opt/fhi-aims/bin/aims.serial.x > aims.dft.out 2>&1'
# run_aims_cmd = 'mpirun -n 4 /home/tianren/Documents/fhi-aims/bin/mpi.mkl.x > aims.dft.out 2>&1'
init_hess_factor = 1.  # the initial Hessian is this times of the AIMS-initialized Hessian
use_aims_prediction = False  # for TRM subproblem, use geometry.in.next_step as the initial guess
default_trust_radius = 0.2  # (Ang) the default trust radius if not otherwise provided
force_criterion = 0.01  # (eV/Ang) the stopping criterion on the force for relaxation
energy_tolerance = 2e-4  # (eV) energy rise bigger than this in a step raises error
interrupt_criteria = 1e-4  # (Ang) the interruption criterion on the update for relaxation
gliding_step_criteria = 2e-4  # (eV) a series of gliding steps have energy change smaller than this
max_gliding_steps = 5  # abort if there are this many continuous accepted gliding steps
max_unaccepted_steps = 8  # abort if there are this many continuous unaccepted steps
max_iterations = 1000  # the max number of optimization iterations

# constants and system variables
OPJ = os.path.join
pwd = os.getcwd()
_re_none_space = re.compile(r'\S+')
epsilon = 1e-10  # small positive number, AIMS default
native_eps = 1.4901161193847656e-08  # sqrt of the float limit
harmonic_length_scale = 0.025  # (Ang) geometry updates smaller than this are harmonic
dist_HH = 0.69  # (Ang) the length of H-H bond, smallest possible distance between atoms


# region math functions
def minimum_distance(coordinate) -> float:
    """
    :param coordinate: 1d array of coordinates of atoms, like [x, y, z, x, y, z, ...].
    :return: the min distance between any two atoms.
    """
    natoms = len(coordinate) // 3
    coordinate = np.reshape(coordinate, newshape=(natoms, 3))
    min_dist = float('Inf')
    for i in range(natoms):
        for j in range(i + 1, natoms):
            dist = np.linalg.norm(coordinate[i] - coordinate[j])
            min_dist = min(min_dist, dist)
    return min_dist


def angle_between(vec1, vec2) -> float:
    """
    quickly calculate the absolute angle between two vectors.
    :param vec1, vec2: two 1d vectors.
    :return: the angle between vec1 and vec2, [0, pi], in radians.
    """
    if len(vec1) in (2, 3):  # cross only works for 2d or 3d
        atan_y = np.linalg.norm(np.cross(vec1, vec2))
        atan_x = np.linalg.norm(vec1) * np.linalg.norm(vec2) + np.dot(vec1, vec2)
        return 2. * np.arctan2(atan_y, atan_x)
    return np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
                             a_min=0., a_max=1.))


def func_dist_2atoms(atom1, atom2) -> typing.Callable[[np.ndarray], float]:
    """
    gives a function to calculate the distance between 2 atoms, in Angstroms.
    :param atom1, atom2: 0-based atom numbers.
    :return: function(coordinate: ndarray[3 * natoms of float]) -> distance in Angstroms.
    """
    a1s, a1e = atom1 * 3, atom1 * 3 + 3
    a2s, a2e = atom2 * 3, atom2 * 3 + 3
    def dist_2atoms(coordinate: np.ndarray) -> float:
        return np.linalg.norm(coordinate[a2s:a2e] - coordinate[a1s:a1e])
    return dist_2atoms


def func_angle_3atoms(atom1, atom2, atom3) -> typing.Callable[[np.ndarray], float]:
    """
    gives a function to calculate the angle between 3 atoms, in radians, range [0, pi].
    :param atom1, atom2, atom3: 0-based atom numbers, atom2 is the center point.
    :return: function(coordinate: ndarray[3 * natoms of float]) -> angle in radians.
    """
    a1s, a1e = atom1 * 3, atom1 * 3 + 3
    a2s, a2e = atom2 * 3, atom2 * 3 + 3
    a3s, a3e = atom3 * 3, atom3 * 3 + 3
    def angle_3atoms(coordinate: np.ndarray) -> float:
        vec1 = coordinate[a1s:a1e] - coordinate[a2s:a2e]  # 2 -> 1
        vec2 = coordinate[a3s:a3e] - coordinate[a2s:a2e]  # 2 -> 3
        atan_y = np.linalg.norm(np.cross(vec1, vec2))
        atan_x = np.linalg.norm(vec1) * np.linalg.norm(vec2) + np.dot(vec1, vec2)
        return 2. * np.arctan2(atan_y, atan_x)
    return angle_3atoms


def func_dihedral_4atoms(atom1, atom2, atom3, atom4,
                         positive_range=False) -> typing.Callable[[np.ndarray], float]:
    """
    gives a function to calculate the dihedral between 4 atoms, in radians, range [-pi, pi].
    :param atom1, atom2, atom3, atom4: 0-based atom numbers, atom2 and atom3 define the hinge.
    :param positive_range: set to True to change range to [0, 2pi).
    :return: function(coordinate: ndarray[3 * natoms of float]) -> dihedral in radians.
    """
    a1s, a1e = atom1 * 3, atom1 * 3 + 3
    a2s, a2e = atom2 * 3, atom2 * 3 + 3
    a3s, a3e = atom3 * 3, atom3 * 3 + 3
    a4s, a4e = atom4 * 3, atom4 * 3 + 3
    def dihedral_4atoms(coordinate: np.ndarray) -> float:
        vec1 = coordinate[a2s:a2e] - coordinate[a1s:a1e]  # 1 -> 2
        vec2 = coordinate[a3s:a3e] - coordinate[a2s:a2e]  # 2 -> 3
        vec3 = coordinate[a4s:a4e] - coordinate[a3s:a3e]  # 3 -> 4
        surf1 = np.cross(vec1, vec2)
        surf2 = np.cross(vec2, vec3)
        cross = np.cross(surf1, surf2)
        atan_y = np.linalg.norm(cross) * np.sign(np.dot(cross, vec2))
        atan_x = np.linalg.norm(surf1) * np.linalg.norm(surf2) + np.dot(surf1, surf2)
        return 2. * np.arctan2(atan_y, atan_x)
    if not positive_range:
        return dihedral_4atoms
    else:
        def wrapped(coordinate: np.ndarray) -> float:
            angle = dihedral_4atoms(coordinate)
            if angle < 0:
                angle += 2. * np.pi
            return angle
        return wrapped
# endregion math functions


# region I/O functions
def _raise_syntax_error(keyword, file_name, line_number=None) -> None:
    """raise syntax error of keyword, specifying the file_name and line_number."""
    line_number = str(line_number)
    if keyword == 'EOF':
        raise EOFError('unexpected end of file in file "%s".' % file_name)
    else:
        raise SyntaxError('"%s" not expected in file "%s" line #%s.' % (keyword, file_name, line_number))


def _resolve_arguments(arguments) -> (float, float):
    """
    resolve arguments such as "50", ">8.9", ">70 <90", etc.
    :param arguments: 1 or 2 words. each word is like [sign]number, only "<", ">" and "=" are allowed
                      for the sign. put in an "=" if no sign is given.
    :return: (lb, ub), lower bound and upper bound.
    """
    lb, ub = -np.inf, np.inf
    for word in arguments:
        # find the number
        if word[0] in ('<', '>', '='):
            number = float(word[1:])
        else:
            number = float(word)
        # find the sign and apply bound
        if word[0] == '<':
            ub = number
        elif word[0] == '>':
            lb = number
        else:  # '=' or by default
            ub = number
            lb = number
    return lb, ub


def _bounds_to_string(lb, ub, decimal_place=2) -> str:
    """convert the bounds (lb, ub) into a grammatical string to print."""
    if lb == -np.inf:
        if ub == np.inf:
            return 'not constraint'
        return 'smaller than %%.%df' % decimal_place % ub
    elif ub == np.inf:
        return 'larger than %%.%df' % decimal_place % lb
    elif lb == ub:
        return 'equal to %%.%df' % decimal_place % lb
    else:
        return 'between %%.%df and %%.%df' % (decimal_place, decimal_place) % (lb, ub)


def read_geometry(file_name='geometry.in') -> (list, np.ndarray):
    """
    read the coordinate of atoms from a "geometry.in" file.
    :param file_name: the path+name to the file to read.
    :return[0]: geo_contents, a list of str stores each line of the "geometry.in" file, but if a
                line is an atom, this line is changed to a dict {number: int, type: str}.
    :return[1]: coordinate, the 1d array of the coordinates of each atom on each axis.
    """
    geo_contents = []
    coordinate = []
    with open(file_name, mode='r') as f:
        atom_number = -1  # starting from 0
        for line in f:
            words = re.findall(_re_none_space, line)
            if (len(words) > 0) and (words[0] == 'atom'):
                atom_number += 1
                atom_type = words[4]
                geo_contents.append({'number': atom_number, 'type': atom_type})
                for i in (0, 1, 2):
                    coordinate.append(float(words[i + 1]))
            elif (len(words) > 0) and (words[0] in ['trust_radius', 'hessian_block']):
                continue  # remove these keywords
            elif len(line) >= 1:
                geo_contents.append(line[:-1])
    coordinate = np.array(coordinate, dtype=float)
    return geo_contents, coordinate


def write_geometry(file_name, geo_contents, coordinate, trust_radius=None, hessian=None) -> None:
    """
    write a "geometry.in.next_step" file.
    :param file_name: the path+name to the file to write the contents into.
    :param geo_contents: a list of str stores each line of the "geometry.in" file, but if a line is
                         an atom, this line is changed to a dict {number: int, type: str}.
    :param coordinate: the 1d array of the coordinates of each atom on each axis.
    :param trust_radius: (optional) the trust radius to print.
    :param hessian: (optional) the hessian matrix to print.
    """
    with open(file_name, mode='w') as f:
        for content in geo_contents:
            if isinstance(content, dict):
                atom_number = content['number']
                atom_type = content['type']
                atom_x, atom_y, atom_z = coordinate[atom_number * 3: atom_number * 3 + 3]
                f.write('atom  %15.8f %15.8f %15.8f %s\n' %
                        (atom_x, atom_y, atom_z, atom_type))
            else:  # isinstance(content, str)
                f.write(content + '\n')
        if trust_radius is not None:
            f.write('trust_radius          %15.10f\n' % trust_radius)
        if hessian is not None:
            natoms = hessian.shape[0] // 3
            for atom_i in range(natoms):
                for atom_j in range(atom_i, natoms):
                    submatrix_str = ''
                    for i in range(atom_i * 3, atom_i * 3 + 3):
                        for j in range(atom_j * 3, atom_j * 3 + 3):
                            submatrix_str += '    %15.6f' % hessian[i][j]
                    f.write('hessian_block          % 7d % 7d%s\n'
                            % (atom_i + 1, atom_j + 1, submatrix_str))


def read_hessian(file_name='geometry.in.next_step') -> (float, np.ndarray):
    """
    read the trust radius and the Hessian matrix from a "geometry.in.next_step" file.
    :param file_name: the path+name to the file to read.
    :return[0]: the trust radius of the TRM (in Ang).
    :return[1]: the analog Hessian matrix of the BFGS method (in eV/Ang^2).
    """
    trust_radius = None
    hessian = None
    with open(file_name, mode='r') as f:
        reading_atoms = True
        natoms = 0
        hessian_line_read = None
        line_number = 0
        for line in f:
            line_number += 1  # line_number starts from 1
            words = re.findall(_re_none_space, line)
            if (len(words) <= 1) or (words[0][0] == '#'):
                continue  # empty or comment line
            elif words[0] == 'atom':
                if not reading_atoms:
                    _raise_syntax_error(words[0], file_name, line_number)
                natoms += 1
            elif words[0] == 'trust_radius':
                if trust_radius is not None:
                    _raise_syntax_error(words[0], file_name, line_number)
                trust_radius = float(words[1])
            elif words[0] == 'hessian_block':
                reading_atoms = False  # must read all atoms before this
                if hessian is None:
                    hessian_line_read = np.zeros(shape=(natoms, natoms), dtype=bool)
                    hessian = np.zeros(shape=(3 * natoms, 3 * natoms), dtype=float)
                atom_i, atom_j = int(words[1]) - 1, int(words[2]) - 1
                if hessian_line_read[atom_i, atom_j]:  # already saw this hessian_block
                    _raise_syntax_error(' '.join(words[0:3]), file_name, line_number)
                hessian_line_read[atom_i, atom_j] = True
                hessian_line_read[atom_j, atom_i] = True
                for i in (0, 1, 2):
                    for j in (0, 1, 2):
                        h = float(words[3 * i + j + 3])
                        ii, jj = atom_i * 3 + i, atom_j * 3 + j
                        hessian[ii, jj] = h
                        hessian[jj, ii] = h
    if trust_radius is None:
        sys.stdout.write('warning: trust_radius is not provided.\n')
    if not np.all(hessian_line_read):  # AIMS does not print if a sub-matrix is zero
        pass  # sys.stdout.write('warning: some hessian_block`s are completely zero.\n')
    if reading_atoms or (hessian is None):
        _raise_syntax_error('EOF', file_name)
    return trust_radius, hessian


def read_force(file_name='aims.dft.out') -> (float, float, np.ndarray):
    """
    read the total energy and the vector of force from an "aims.dft.out" file. read the first
    occurrence in the file only.
    :param file_name: the path+name to the file to read.
    :return[0]: the total energy of the geometry (in eV).
    :return[1]: the electronic free energy of the geometry (in eV)
    :return[2]: the cleaned force on each axes of atoms of the geometry (in eV/Ang).
    """
    natoms_line_start = re.compile(r'  \| Number of atoms')
    energy_line_start = re.compile(r'  \| Total energy uncorrected      :')
    free_energy_line_start = re.compile(r'  \| Electronic free energy        :')
    # force_title_line = re.compile(r'  Total atomic forces \(unitary forces cleaned\) \[eV\/Ang\]:')
    force_title_line = re.compile(r'  Total atomic forces \(.*\) \[eV\/Ang\]:')
    number_pattern = re.compile(r'[-+]?\d*\.?\d+[eE]?[-+]?\d*')
    energy = None
    free_energy = None
    force = None
    with open(file_name, mode='r') as f:
        natoms = None
        reading_force = False
        force_line_read = None
        line_number = 0
        for line in f:
            line_number += 1  # line number starts from 1
            if reading_force:
                if len(line) <= 1:
                    reading_force = False
                    continue  # always there is an empty line after printed force
                if force_line_read is None:
                    if natoms is None:
                        _raise_syntax_error('Total atomic forces', file_name, line_number)
                    force_line_read = np.zeros(shape=(natoms,), dtype=bool)
                    force = np.zeros(shape=(3 * natoms,), dtype=float)
                words = re.findall(_re_none_space, line)
                atom = int(words[1]) - 1
                if not force_line_read[atom]:  # the first occurrence
                    force_line_read[atom] = True
                    for i in (0, 1, 2):
                        f = float(words[i + 2])
                        ii = atom * 3 + i
                        force[ii] = f
            elif re.match(force_title_line, line):
                reading_force = True
            elif re.match(natoms_line_start, line):
                if natoms is not None:  # should be only one line like this
                    _raise_syntax_error('Number of atoms', file_name, line_number)
                number = re.findall(number_pattern, line)[0]
                natoms = int(number)
            elif re.match(energy_line_start, line):
                if energy is None:  # the first occurrence
                    number = re.findall(number_pattern, line)[0]
                    energy = float(number)
            elif re.match(free_energy_line_start, line):
                if free_energy is None:  # the first occurrence
                    numbers = re.findall(number_pattern, line)
                    if len(numbers) == 1:  # there are two kinds of this sentences... only want this
                        free_energy = float(numbers[0])
    if reading_force or (force is None) or (not np.all(force_line_read)):
        _raise_syntax_error('EOF', file_name)
    return energy, free_energy, force


def read_and_check_control(file_name='control.in', modify=True, max_relaxation_steps=0) -> float:
    """
    read the force criteria from a "control.in" file, and make sure it contains "relax_geometry" and
    set "max_relaxation_steps" keyword to "1".
    :param file_name: the path+name to the file to read and check.
    :param modify: set to False to avoid making any change in the file, if so "check" won't be done.
    :param max_relaxation_steps: if modify, the desired max_relaxation_steps to write in.
    :return: the force criteria when the relaxation stops, in eV/Ang, is 0.01 by default.
    """
    force_criteria = 0.01  # (eV/Ang)
    # read everything in the file
    with open(file_name, mode='r') as f:
        lines = f.readlines()
    # then write everything back
    with open(file_name, mode='w') as f:
        relax_geometry_set = False
        exceptions_caught = []
        # print lines read
        for line in lines:
            words = re.findall(_re_none_space, line)
            if (len(words) > 0) and (words[0] == 'relax_geometry'):
                try:
                    f.write(line)
                    force_criteria = float(words[2])
                    relax_geometry_set = True
                    if modify:  # always print this statement here
                        f.write('  max_relaxation_steps  %d\n' % max_relaxation_steps)
                except Exception as e:
                    exceptions_caught.append(e)
            elif modify and (len(words) > 0) and (words[0] == 'max_relaxation_steps'):
                continue  # remove because we have printed this line
            else:
                try:
                    f.write(line)
                except Exception as e:
                    exceptions_caught.append(e)
        # print "relax_geometry" at the end if not found
        if (relax_geometry_set is False) and modify:
            try:
                f.write('  relax_geometry   bfgs %f\n' % force_criteria)
                f.write('  max_relaxation_steps  %d\n' % max_relaxation_steps)
            except Exception as e:
                exceptions_caught.append(e)
    # deal with exceptions
    if len(exceptions_caught) > 0:
        if len(exceptions_caught) == 1:
            raise exceptions_caught[0]
        else:
            raise Exception(exceptions_caught)
    # return
    return force_criteria


def get_and_put_keyword(file_name, keyword, put=None) -> typing.Union[str, None]:
    """
    find the first occurrence of a keyword in a file, return the string following the keyword
    (separated by white space). if "put" is set, rewrite this string in the file into "put",
    rewrite the first occurrence only, too.
    :param file_name: the path+name to the file to read and rewrite.
    :param keyword: the first word of the line to look for.
    :param put: set to some string to rewrite, leave to be None if not modifying.
    :return: the string following the keyword if the keyword is found.
    """
    # read everything in the file
    with open(file_name, mode='r') as f:
        lines = f.readlines()
    # find the string to return
    to_return = None
    for line in lines:
        words = re.findall(_re_none_space, line)
        if (len(words) > 0) and (words[0] == keyword):
            if len(words) > 1:
                to_return = ' '.join(words[1:])
            else:
                to_return = ''
            break
    # write back if put is set
    if put is None:
        return to_return
    exceptions_caught = []
    already_writen = False
    with open(file_name, mode='w') as f:
        # print lines read
        for line in lines:
            words = re.findall(_re_none_space, line)
            if (not already_writen) and (len(words) > 0) and (words[0] == keyword):
                try:
                    f.write(keyword + ' ' + put + '\n')
                except Exception as e:
                    exceptions_caught.append(e)
                already_writen = True
            else:
                try:
                    f.write(line)
                except Exception as e:
                    exceptions_caught.append(e)
        # print at the end if keyword not found
        if not already_writen:
            try:
                f.write(keyword + ' ' + put + '\n')
            except Exception as e:
                exceptions_caught.append(e)
    # deal with exceptions
    if len(exceptions_caught) > 0:
        if len(exceptions_caught) == 1:
            raise exceptions_caught[0]
        else:
            raise Exception(exceptions_caught)
    # return
    return to_return


def read_bounds(coordinate, file_name='constraints.in') -> typing.Union[opt.Bounds, None]:
    """
    read the bounds from a "constraints.in" file, including keywords "fix", "x", "y" and "z".
    notice this function does NOT check for index out of bound, so it may crash in opt.minimize().
    :param coordinate: the np.ndarray of initial geometry, used for omitted arguments.
    :param file_name: the path+name to the file to read.
    :return: the opt.Bounds object to apply in opt.minimize(). None if no bounds is set.
    """
    natoms = coordinate.shape[0] // 3
    lb = np.full([natoms * 3], fill_value=-np.inf, dtype=float)
    ub = np.full([natoms * 3], fill_value=np.inf, dtype=float)
    no_bounds_set = True
    with open(file_name, mode='r') as f:
        for line in f:
            words = re.findall(_re_none_space, line)
            if len(words) < 2:  # not a valid command
                continue
            if words[0].lower() == 'fix':
                atom = int(words[1]) - 1  # 1-based in file, but 0-based in program
                ix = atom * 3  # just to be short..., index of x of atom
                for i in (0, 1, 2):  # x, y, z
                    lb[ix + i] = coordinate[ix + i]
                    ub[ix + i] = coordinate[ix + i]
                sys.stdout.write('fixing the position of atom %d' % (atom + 1))
                sys.stdout.write(' (now [%.2f  %.2f  %.2f] in Ang)\n' %
                                 (coordinate[ix], coordinate[ix + 1], coordinate[ix + 2]))
                no_bounds_set = False
            elif words[0].lower() in ('x', 'y', 'z'):  # x, y or z
                atom = int(words[1]) - 1  # 1-based in file, but 0-based in program
                w0 = words[0].lower()
                i = 0 if (w0 == 'x') else (1 if (w0 == 'y') else 2)  # x/y/z -> 0/1/2
                l = u = coordinate[atom * 3 + i]
                if len(words) > 2:  # arguments provided
                    l, u = _resolve_arguments(words[2:])
                lb[atom * 3 + i] = l
                ub[atom * 3 + i] = u
                sys.stdout.write('fixing the %s component of atom %d to be %s in Ang' %
                                 (words[0].lower(), atom + 1, _bounds_to_string(l, u)))
                sys.stdout.write(' (now %.2f)\n' % coordinate[atom * 3 + i])
                no_bounds_set = False
    # wrap returning object
    if no_bounds_set:
        return None
    else:
        return opt.Bounds(lb=lb, ub=ub)


def read_constraints(coordinate, file_name='constraints.in') -> list:
    """
    read constraints from a "constraints.in" file, including keywords "bond", "angle" and "dihedral".
    notice this function does NOT check for index out of bound, so it may crash in opt.minimize().
    :param coordinate: the np.ndarray of initial geometry, used for omitted arguments.
    :param file_name: the path+name to the file to read.
    :return: list of constraints to apply in opt.minimize().
    """
    natoms = coordinate.shape[0] // 3
    constraints = []
    small_rad = 30. * np.pi / 180.  # some angle close to +/- pi will trigger round guard
    # sub-functions used in the process
    def deg2rad(_deg):  # transform degrees into radians, and limit to [-pi, pi)
        if (_deg == np.inf) or (_deg == -np.inf):
            return _deg
        _rad = _deg * np.pi / 180.
        while _rad >= np.pi:
            _rad -= 2. * np.pi
        while _rad < -np.pi:
            _rad += 2. * np.pi
        return _rad
    # start reading
    with open(file_name, mode='r') as f:
        line_number = 0  # starting from 1
        for line in f:
            line_number += 1
            words = re.findall(_re_none_space, line)
            if len(words) < 2:  # not a valid command
                continue
            # keyword "bond"
            if words[0].lower() == 'bond':
                atom1 = int(words[1]) - 1  # 1-based in file, but 0-based in program
                atom2 = int(words[2]) - 1  # 1-based in file, but 0-based in program
                fun = func_dist_2atoms(atom1, atom2)
                if len(words) == 3:  # no argument is provided
                    lb = ub = fun(coordinate)
                else:
                    lb, ub = _resolve_arguments(words[3:])
                constraints.append(opt.NonlinearConstraint(
                    fun=fun, lb=lb, ub=ub,
                ))
                sys.stdout.write('fixing the bond %d-%d to be %s Ang' %
                                 (atom1 + 1, atom2 + 1, _bounds_to_string(lb, ub)))
                sys.stdout.write(' (now %.2f)\n' % fun(coordinate))
            ################################################################
            # keyword "angle" - range [0, pi]
            # the angles in [-pi, 0) will be reflected into (0, pi]
            elif words[0].lower() == 'angle':
                atom1 = int(words[1]) - 1  # 1-based in file, but 0-based in program
                atom2 = int(words[2]) - 1  # 1-based in file, but 0-based in program
                atom3 = int(words[3]) - 1  # 1-based in file, but 0-based in program
                fun = func_angle_3atoms(atom1, atom2, atom3)
                if len(words) == 4:  # no argument is provided
                    lb = ub = fun(coordinate)
                else:
                    lb, ub = _resolve_arguments(words[4:])  # in degrees
                    lb = deg2rad(lb)
                    ub = deg2rad(ub)
                lb = 0. if (lb == -np.inf) else lb  # need to set -inf to 0. for processing
                if (lb < 0.) and (ub > 0.):  # wrapping around 0 deg
                    ub = max(-lb, ub)
                    lb = -np.inf
                elif ub < lb:  # wrapping around 180 deg
                    lb = min(lb, -ub)
                    ub = np.inf
                else:  # regular reflection
                    lb = -lb if (lb < 0.) else lb  # [0, pi]
                    ub = -ub if (ub < 0.) else ub  # [0, pi]
                    if ub < lb:
                        lb, ub = ub, lb
                lb = -np.inf if (lb <= 0.) else lb  # 0. means no constrain
                ub = np.inf if (ub >= np.pi) else ub  # pi means no constrain
                if ((lb == -np.inf) and (ub == np.inf)) or (ub < lb):
                    sys.stdout.write('warning: in "%s", line %d is invalid\n' %
                                     (file_name, line_number))
                    continue  # bad input or no need to constrain
                constraints.append(opt.NonlinearConstraint(
                    fun=fun, lb=lb, ub=ub,
                ))
                sys.stdout.write('fixing the angle %d-%d-%d to be %s deg' %
                                 (atom1 + 1, atom2 + 1, atom3 + 1,
                                  _bounds_to_string(lb * 180. / np.pi, ub * 180. / np.pi, 1)))
                sys.stdout.write(' (now %.1f)\n' % (fun(coordinate) * 180. / np.pi))
            ################################################################
            # keyword "dihedral" - range [-pi, pi]
            # round guard is applied: when close to bounds, set range to [0, 2pi)
            elif words[0].lower() == 'dihedral':
                atom1 = int(words[1]) - 1  # 1-based in file, but 0-based in program
                atom2 = int(words[2]) - 1  # 1-based in file, but 0-based in program
                atom3 = int(words[3]) - 1  # 1-based in file, but 0-based in program
                atom4 = int(words[4]) - 1  # 1-based in file, but 0-based in program
                fun = func_dihedral_4atoms(atom1, atom2, atom3, atom4)
                if len(words) == 5:  # no argument is provided
                    lb = ub = fun(coordinate)
                else:
                    lb, ub = _resolve_arguments(words[5:])  # in degrees
                    lb = deg2rad(lb)
                    ub = deg2rad(ub)
                triggered = [  # round guard triggers
                    (ub < lb),  # one bound goes around
                    (lb < -np.pi + small_rad and ub < 0. - small_rad),  # close to -pi
                    (ub > np.pi - small_rad and lb > 0. + small_rad)  # close to +pi
                ]
                if any(triggered):  # round guard triggered, set range to [0, 2pi)
                    lb = (lb + 2. * np.pi) if (lb < 0.) else lb
                    ub = (ub + 2. * np.pi) if (ub < 0.) else ub
                    fun = func_dihedral_4atoms(atom1, atom2, atom3, atom4, positive_range=True)
                if (lb == -np.inf) or (ub == np.inf) or (ub < lb):  # both lb and ub need to exist
                    sys.stdout.write('warning: in "%s", line %d is invalid\n' %
                                     (file_name, line_number))
                    continue  # bad input or no need to constrain
                constraints.append(opt.NonlinearConstraint(
                    fun=fun, lb=lb, ub=ub,
                ))
                sys.stdout.write('fixing the dihedral %d-%d-%d-%d to be %s deg' %
                                 (atom1 + 1, atom2 + 1, atom3 + 1, atom4 + 1,
                                  _bounds_to_string(lb * 180. / np.pi, ub * 180. / np.pi, 1)))
                sys.stdout.write(' (now %.1f)\n' % (fun(coordinate) * 180. / np.pi))
    return constraints


def write_relaxation_info(file_name, energy, free_energy, max_force,
                          accepted, is_final=False) -> None:
    """
    write or append the energy and force information to a "relaxation_info.dat" file.
    :param file_name: the path+name to the file to write or append to.
    :param energy: the total energy in eV.
    :param free_energy: the electron free energy in eV.
    :param max_force: the max force on any coordinate components in eV/Ang.
    :param accepted: whether this step is accepted or not.
    :param is_final: whether this is the final geometry or not.
    """
    if not os.path.isfile(file_name):
        with open(file_name, mode='w') as f:
            # write the title line
            f.write('\n# Step Total energy [eV]   E-E(1) [meV]   Free energy [eV]'
                    '   F-F(1) [meV]   max. force [eV/AA]\n \n')
            # write the first line
            f.write('%5i   %16.8f %14.6f   %16.8f %14.6f %10.6f \n' %
                    (1, energy, 0., free_energy, 0., max_force))
    else:  # file exists
        # read the step number, E(1) and F(1) from the file
        with open(file_name, mode='r') as f:
            lines = f.readlines()
            first_line = lines[3]  # empty; title; empty; first...
            words = re.findall(_re_none_space, first_line)
            energy_1 = float(words[1])
            free_energy_1 = float(words[3])
            nsteps = len(lines) - 3  # 2 empty lines and 1 title line
        # append to the end of file
        with open(file_name, mode='a') as f:
            f.write('%s %16.8f %14.6f   %16.8f %14.6f %10.6f %s\n' %
                    ('finally' if is_final else ('%5d  ' % (nsteps + 1)),
                     energy, energy - energy_1,
                     free_energy, free_energy - free_energy_1,
                     max_force,
                     '' if accepted else ' rejected. '))


def write_xyz_movie(file_name='xyz_movie.dat', workspace='.', mask=None) -> None:
    """
    write a "xyz_movie.dat" file, according to the geometries in workspace/folders. if mask is
    provided, the rejected steps are skipped, as workspace/i+1 is skipped if mask[i] is False.
    :param file_name: the path+name to the file to write the xyz movie into.
    :param workspace: the folders of DFT calculations are under this directory.
    :param mask: (optional) list of bool, whether a step is accepted, default is all True.
    """
    # write list of atoms in one structure
    def write_atoms(_f, _contents, _coord):
        for _l in _contents:
            if isinstance(_l, dict):
                _iatom = _l['number']
                _f.write('% 2s' % _l['type'])
                for i in (0, 1, 2):
                    _f.write(' % 16.6f' % _coord[_iatom * 3 + i])
                _f.write('\n')
    # start process
    with open(file_name, mode='w') as f:
        # write "geometry.in" under each folder in
        istep = 0  # starting from 0
        while os.path.isdir(OPJ(workspace, str(istep))):
            if (mask is None) or (istep == 0) or mask[istep - 1]:
                geo_contents, coordinate = read_geometry(OPJ(workspace, str(istep), 'geometry.in'))
                # write title
                f.write('% 12d\n' % (coordinate.shape[0] // 3))
                f.write('Iteration:% 6d\n' % istep)
                # write atoms
                write_atoms(f, geo_contents, coordinate)
            istep += 1
        # finally write "geometry.in.next_step"
        if True:  # (mask is None) or mask[-1]:
            geo_contents, coordinate = read_geometry(OPJ(workspace, 'geometry.in.next_step'))
            # write title
            f.write('% 12d\n' % (coordinate.shape[0] // 3))
            f.write('Final Geometry\n')
            # write atoms
            write_atoms(f, geo_contents, coordinate)
# endregion I/O functions


# region relaxation
class _State:
    """struct that stores the state of one iteration step of relaxation."""
    def __init__(self):
        self.X = None  # [3 * natoms], the coordinate
        self.E = None  # scalar, the total energy
        self.F = None  # [3 * natoms], the force
        self.H = None  # [3 * natoms, 3 * natoms], the Hessian matrix
        self.D = None  # [3 * natoms], the search direction
        self.r = None  # scalar, the trust radius
        self.dX = None  # [3 * natoms], the update of X, X[next] <- X + dX
        self.accepted = False  # bool, whether this step is accepted or not
        self.free_E = None  # scalar, the electronic free energy


class NotConvergedException(Exception):
    """raise this if relaxation is aborted but not converged."""
    pass


class Relaxation:
    """workspace of one geometry relaxation."""
    def __init__(self):
        self.run_aims_cmd = run_aims_cmd  # the shell command to run aims
        self.force_criterion = force_criterion  # (eV/Ang) converge if max force is smaller than this
        self.bounds = None  # custom bounds on relaxation
        self.constraints = None  # custom constrains on relaxation
        self.not_converged = True  # whether the relaxation needs to be continued
        self.last_state = None  # the last accepted _State
        self.best_state = None  # the _State with lowest energy
        # self.last_feasible_hess = None  # the last accepted Hessian, or the initial one
        self.geo_contents = None  # the contents in "geometry.in" file
        self.energies = []  # the total energies for each iteration
        self.free_energies = []  # the electric free energies for each iteration
        self.max_forces = []  # the max force components for each iteration
        self.steps_accepted = []  # for each step, whether it is accepted or not
        self.angles_to_forces = []  # the angle between dX and F for each iteration

    @classmethod
    def BFGS_advance(cls, state, last_state, assure_posdef=True) -> None:
        """
        given last state and this state, calculate the Hessian for this state by BFGS method.
        :param state: this state, F must be given. H will be updated here inplace.
        :param last_state: last state, H, F and dX must be given.
        :param assure_posdef: set to true to assure the updated H is positive definite.
        """
        Y = last_state.F - state.F
        S = last_state.dX
        if assure_posdef:
            R = (1. + max(0., -np.dot(Y, S) / np.dot(S, S))) * np.linalg.norm(last_state.F)
            Y = Y + R * S  # to assure that H is positive definite
        U = np.reshape(Y, newshape=(-1, 1))
        V = np.reshape(np.dot(last_state.H, S), newshape=(-1, 1))
        alpha = 1. / np.dot(Y, last_state.dX)
        beta = -1. / np.dot(S, np.dot(last_state.H, S))
        state.H = last_state.H + alpha * np.matmul(U, U.T) + beta * np.matmul(V, V.T)

    @classmethod
    def TRM_advance(cls, state, r_eff=None, H_eff=None, no_hess=False,
                    bounds=None, constraints=None) -> None:
        """
        given a state, calculate the step update dX for this state by TRM.
        :param state: X, F, H, r must be given. D and dX will be updated here inplace.
        :param r_eff: provide this to override state.r only for this time.
        :param H_eff: provide this to override state.H only for this time.
        :param no_hess: set to true to ignore Hessian in TRM subproblem.
        :param bounds: the opt.Bounds to be applied on the TRM subproblem.
        :param constraints: the list of constraints to be applied on the TRM subproblem.
        """
        # rationalize arguments
        r_eff = state.r if (r_eff is None) else r_eff
        H_eff = state.H if (H_eff is None) else H_eff
        constraints = [] if (constraints is None) else constraints
        constraints = [constraints] if (not isinstance(constraints, (list, tuple))) else constraints
        # duplicate const mutable arguments
        constraints = [x for x in constraints]
        # find initial guesses
        if state.D is None:
            state.D = np.linalg.solve(H_eff, state.F)  # search direction by Newton
        if np.linalg.norm(state.D) > r_eff:
            state.D = state.D * r_eff / (np.linalg.norm(state.D) + epsilon)
        # setup TRM radius constraint
        min_eigval = np.real(min(np.linalg.eig(H_eff)[0]))  # eig(M) returns [M.eigval, M.eigvec]
        if min_eigval < -1000.:  # todo: is this correct????
            if not no_hess:
                sys.stdout.write('warning: broken Hessian, using grad-only\n')
            no_hess = True  # todo: is this correct????
        radius_fixed = (min_eigval < epsilon) and (not no_hess) \
                       and (bounds is None) and (len(constraints) == 0)  # todo: is this true?
        r_eff_2 = r_eff ** 2
        radius_constraint = opt.NonlinearConstraint(
            fun=lambda x: np.linalg.norm(x - state.X) ** 2,  # = <dX|dX>
            lb=r_eff_2 if radius_fixed else -np.inf, ub=r_eff_2,
            jac=lambda x: 2. * (x - state.X),  # = 2|dX>
        )
        constraints.append(radius_constraint)
        # setup TRM subproblem
        # note that the x in lambda expressions denotes to (X + dX)
        # dX = arcmin -<F|dX> + 0.5 * <dX|H|dX>, s.t. <dX|dX> <= r_eff^2
        # is equivalent to this --
        # TRM-sub:  x = arcmin -<F|x> + 0.5 * <x|H|x> - <x|H|X>
        # s.t.   :  <x - X|x - X> <= r_eff^2
        subproblem = lambda x: -np.dot(state.F, x) + 0.5 * np.dot(x, np.dot(H_eff, x - 2. * state.X))
        subproblem_grad = lambda x: -state.F + np.dot(H_eff, x - state.X)
        # subproblem_hess = lambda x: H_eff
        if no_hess:  # assume H = 0 here
            subproblem = lambda x: -np.dot(state.F, x)
            subproblem_grad = lambda x: -state.F
            # zeros_like_H = np.zeros_like(H_eff)
            # subproblem_hess = lambda x: zeros_like_H
        # solve TRM subproblem
        result = opt.minimize(
            fun=subproblem, x0=(state.X + state.D),
            jac=subproblem_grad,
            # hess=subproblem_hess,
            bounds=bounds, constraints=constraints,
            method='SLSQP', options={'maxiter': 2000},
        )
        state.dX = result.x - state.X
        # log
        sys.stdout.write('(grad-only) ' if no_hess else '')
        sys.stdout.write('(radius-fixed) ' if radius_fixed else '')
        s = ('%d steps' % result.nit) if (result.nit != 1) else '1 step'
        sys.stdout.write('TRM-subproblem: %s in %s\n' % (result.message, s))
        if np.real(min_eigval) < epsilon:  # maybe complex numerically asymmetric
            sys.stdout.write('min(H.eigval) = %8.5g ,  is not positive\n' % min_eigval)
        sys.stdout.write('||dX|| = %8.5g Ang ,  ' % np.linalg.norm(state.dX))
        sys.stdout.write('||D|| = %8.5g Ang ,  ' % np.linalg.norm(state.D))
        sys.stdout.write('r = %8.5g Ang\n' % r_eff)
        sys.stdout.write('angle<dX,D> = %8.5g deg ,  ' % np.degrees(angle_between(state.dX, state.D)))
        sys.stdout.write('angle<dX,F> = %8.5g deg\n' % np.degrees(angle_between(state.dX, state.F)))
        sys.stdout.write('TRM-sub(dX) = %10.7g eV ,  ' % subproblem(state.dX))
        sys.stdout.write('TRM-sub(D) = %10.7g eV\n' % subproblem(state.D))
        # abort if not converged
        if not result.success:
            raise NotConvergedException(
                'reason: numerical optimizer with constraints cannot converge.\n'
                'please check if there are constraints incompatible with each other.'
            )

    @classmethod
    def avoid_atom_collision(cls, state, *args, **kwargs) -> None:
        """
        given the TRM-solved state, judge whether after the geometry advance, the smallest distance
        between atoms is too small, that the trust radius needs to be reduced to avoid collision. if
        so, redo TRM advance with reduced effective trust radius.
        :param state: this state, X, F, H, r, dX must be given. D and dX may be updated here inplace.
        :param args, kwargs: the arguments to be passed to TRM_advance().
        """
        r_adjust = default_trust_radius  # (Ang) a conservative trust radius
        if state.r <= r_adjust:
            return  # have been using a more conservative radius
        # check if there are atoms too close to each other
        min_dist_updated = minimum_distance(state.X + state.dX)
        adjust_needed = False
        if min_dist_updated < dist_HH:
            min_dist_starting = minimum_distance(state.X)
            adjust_needed = (min_dist_updated < min_dist_starting - 0.3)
        # redo TRM advance with smaller r_eff if there are atoms too close
        if adjust_needed:
            sys.stdout.write('atoms too close, using %f Ang as trust radius.\n' % r_adjust)
            cls.TRM_advance(state, r_adjust, *args, **kwargs)

    @classmethod
    def continuous_accepted_gliding_steps(cls, angles, energies, steps_accepted) -> int:
        """
        count how many continuous accepted gliding steps are there (from the most recent step), not
        counting this step. "gliding" means the angle between dX and F is not changing, while the
        energy is not changing, either.
        the energy of this state (means the last update) and the angle of last state (means the
        last update) are used as benchmark.
        here this function assumes the lengths of "energies" and "step_accepted" to be the same,
        and will check if "angles" is shorter by 1 - this is allowed and this means this function
        is called before "angles" is updated - anyway the most recent angle will not be used. Thus,
        the length of "angles" can be either the same as "energies" or shorter by 1.
        :param angles: list of the angles between dX and F - it refers to this update.
        :param energies: list of energies - it refers to last update.
        :param steps_accepted: list of bool, if this step is accepted - it refers to last update.
        :return: number of continuous accepted gliding steps starting from the most recent step.
        """
        small_angle = 0.75 * np.pi / 180.  # +/- 0.75 deg is regarded as equal
        small_energy = gliding_step_criteria / 2.  # +/- 0.1 meV is regarded as equal
        # check list length and initialize iterator indices
        if (len(energies) <= 1) or (len(angles) <= 1):
            return 0  # no comparison can be made
        istep = 0  # starting from -1
        istep_angles = 0  # starting from -1 or -2, see below
        if len(angles) == len(energies):  # the last angle is updated, but it won't be used
            istep_angles = 1  # then need to start from -2
        # initialize benchmarks
        while istep_angles + 1 < len(angles):  # angles may be shorter
            istep += 1
            istep_angles += 1
            if steps_accepted[-istep]:
                energy_benchmark = energies[-istep]
                angle_benchmark = angles[-istep_angles]
                equal_energy = lambda x: \
                    (energy_benchmark - small_energy < x < energy_benchmark + small_energy)
                equal_angle = lambda x: \
                    (angle_benchmark - small_angle < x < angle_benchmark + small_angle)
                break
        else:
            return 0  # no accepted step is found
        # start counting
        count = 1  # including the benchmark step
        while istep_angles + 1 < len(angles):  # angles may be shorter
            istep += 1
            istep_angles += 1
            if steps_accepted[-istep]:
                if equal_energy(energies[-istep]) and equal_angle(angles[-istep_angles]):
                    count += 1
                else:
                    break  # now it is not continuous
        return count

    @classmethod
    def continuous_unaccepted_steps(cls, steps_accepted) -> int:
        """
        count how many continuous unaccepted steps are there (from the most recent step),
        including this step.
        :param steps_accepted: list of bool, if this step is accepted - it refers to last update.
        :return: number of continuous unaccepted steps starting from the most recent step.
        """
        count = 0
        for i in range(1, len(steps_accepted) + 1):
            if steps_accepted[-i]:
                break
            else:
                count += 1
        return count

    @classmethod
    def TRM_update_radius(cls, state, last_state) -> None:
        """
        given last state and this state, update the trust radius and judge if this step is accepted.
        :param state: this state, E and F must be given. r and accepted will be updated here inplace.
        :param last_state: last state, r, E, F and dX must be given.
        """
        dX_last_norm = np.linalg.norm(last_state.dX)
        ared = state.E - last_state.E  # actual energy gain
        pred_qnm = -np.dot(last_state.F, last_state.dX) \
                   + 0.5 * np.dot(last_state.dX, np.dot(last_state.H, last_state.dX))  # Newton gain
        pred = min(pred_qnm, -epsilon)  # limit pred to be negative
        harmonic = -np.dot(last_state.dX, 0.5 * (last_state.F + state.F))  # harmonic energy gain
        # calculate quality ratio and judge if this step is accepted
        if dX_last_norm <= harmonic_length_scale:
            quality_ratio = harmonic / pred
            state.accepted = (quality_ratio > 0.1)  # more conservative
        else:
            quality_ratio = ared / pred
            state.accepted = (quality_ratio > 0.)
        # log
        sys.stdout.write('E - E[last] = %10.7g eV\n' % ared)
        sys.stdout.write('dE<predict> = %10.7g eV' % pred_qnm)
        sys.stdout.write(' ,  dE<harmonic> = %10.7g eV\n' % harmonic)
        sys.stdout.write('harmonic ' if (dX_last_norm <= harmonic_length_scale) else '')
        sys.stdout.write('quality ratio = %8.5g\n' % quality_ratio)
        # abort if step is small but energy goes up abnormally
        if state.accepted and (ared > energy_tolerance):
            print('warning: relaxation step is small but energy goes up abnormally!')  # todo: Ahhhh!
            state.accepted = False  # todo: is this ok???
            quality_ratio = 0.
        if max(np.abs(last_state.dX)) < interrupt_criteria:  # todo: is this correct?
            raise NotConvergedException(
                'reason: relaxation step is too small. geometry may be converged?\n'
                'maybe this means there is some issue with the s.c.f.-calculated energy surface.\n'
                'you may try to restart the relaxation from this most recent geometry.'
            )
        # update the trust radius
        if quality_ratio < 0.25:  # poor quadraticity
            state.r = min(0.25 * last_state.r, 0.5 * dX_last_norm)
        elif (quality_ratio > 0.75) and (dX_last_norm > 0.8 * last_state.r):  # strong quadraticity
            if dX_last_norm < 0.5 * default_trust_radius:
                state.r = 3. * dX_last_norm
            else:
                state.r = 2. * dX_last_norm
        else:
            state.r = last_state.r

    @classmethod
    def pre_relaxation(cls, state, bounds=None, constraints=None) -> None:
        """
        try to relax the X in the state to satisfy the bounds and constraints.
        :param state: X must be given. X will be updated here inplace.
        :param bounds: the opt.Bounds to be applied on the relaxation.
        :param constraints: the list of constraints to be applied on the relaxation.
        """
        # rationalize arguments
        constraints = [] if (constraints is None) else constraints
        constraints = [constraints] if (not isinstance(constraints, (list, tuple))) else constraints
        constraints = [x for x in constraints]  # duplicate const mutable arguments
        # quit if no constraints are set
        if (bounds is None) and (len(constraints) == 0):
            return
        # quit if all constraints are satisfied
        bounds_tol = 5. * native_eps  # (Ang) allowed error for equality in bounds and constraints
        for _ in (1,):
            if bounds:
                if not np.all(state.X < (bounds.ub + bounds_tol)):
                    break  # not all satisfied
                if not np.all(state.X > (bounds.lb - bounds_tol)):
                    break  # not all satisfied
            for nlc in constraints:
                gx = nlc.fun(state.X)
                if (gx > nlc.ub + bounds_tol) or (gx < nlc.lb - bounds_tol):
                    break  # not all satisfied
            else:
                continue  # all satisfied
            break  # not all satisfied
        else:  # all satisfied
            sys.stdout.write('initial geometry meets all constraints\n')
            return
        # setup and solve pre-relaxation problem
        # note that the x in lambda expressions denotes to (X + dX)
        # --- # problem:  x = arcmin <dX|dX> = arcmin ||x - X||^2
        # --- subproblem = lambda x: np.linalg.norm(x - state.X) ** 2  # = <dX|dX>
        # --- subproblem_grad = lambda x: 2. * (x - state.X)  # = 2|dX>
        # problem:  x = arcmin (<dX|dX>)^2 = arcmin ||x - X||^4
        problem = lambda x: np.linalg.norm(x - state.X) ** 4
        problem_grad = lambda x: 4. * (np.linalg.norm(x - state.X) ** 2) * (x - state.X)
        result = opt.minimize(
            fun=problem, x0=state.X,
            jac=problem_grad,
            bounds=bounds, constraints=constraints,
            method='SLSQP', options={'maxiter': 2000},
        )
        s = ('%d steps' % result.nit) if (result.nit != 1) else '1 step'
        sys.stdout.write('pre-relaxation: %s in %s\n' % (result.message, s))
        # abort if not converged
        if not result.success:
            raise NotConvergedException(
                'reason: numerical optimizer with constraints cannot converge.\n'
                'please check if there are constraints incompatible with each other.'
            )
        # apply pre-relaxation result
        delta = np.linalg.norm(result.x - state.X)
        if delta > default_trust_radius:
            sys.stdout.write('warning: abnormally big move in the pre-relaxation.\n')
            sys.stdout.write('         ||X<pre-relaxed> - X<init>|| = %8.5g Ang\n' % delta)
            sys.stdout.write('         you may need to check the initial geometry.\n')
        if delta > 0.01 * (state.r if state.r else default_trust_radius):
            state.X = result.x  # use pre-relaxed geometry if change > 1% * trust radius
            sys.stdout.write('initial geometry is modified to meet constraints\n')

    @classmethod
    def force_constraint_adjusted(cls, state, bounds=None, constraints=None) -> np.ndarray:
        """
        based on the X and F in the provided state, calculate the constraint-adjust F.
        the constraint-adjusting is that if some components in the original F can be cancelled by
        some constraints, remove these components as well as possible.
        :param state: X and F must be given.
        :param bounds: the opt.Bounds to be applied on the relaxation.
        :param constraints: the list of constraints to be applied on the relaxation.
        :return: the constraint-adjusted F.
        """
        grad_eps = native_eps  # (Ang) a small positive number for gradient calculation here
        bounds_tol = 5. * native_eps  # (Ang) allowed error for equality in bounds and constraints
        # rationalize arguments
        constraints = [] if (constraints is None) else constraints
        constraints = [constraints] if (not isinstance(constraints, (list, tuple))) else constraints
        constraints = [x for x in constraints]  # duplicate const mutable arguments
        # quit if no constraints are set
        if (bounds is None) and (len(constraints) == 0):
            return np.copy(state.F)
        # setup F-adjust problem
        # for the i-th constraint g[i], denote its gradient at state.X as gg[i]
        # we have lambda = arcmin || -F_adj ||    (here we use 2-norm)
        # where -F_adj = -F + lambda[i] * gg[i] + ...
        # if the ub of g[i] is reached, then lambda[i] can be positive
        # if the lb of g[i] is reached, then lambda[i] can be negative
        ggs = []  # the list of gradients of constraints
        ubs = []  # the list of upper bounds of the factor lambda
        lbs = []  # the list of lower bounds of the factor lambda
        # check bounds
        if bounds:
            naxes = state.X.shape[0]
            ub_reached = state.X > bounds.ub - bounds_tol
            lb_reached = state.X < bounds.lb + bounds_tol
            for i in range(naxes):
                if (not ub_reached[i]) and (not lb_reached[i]):
                    continue  # bounds not reached
                ggx = np.zeros(shape=(naxes,), dtype=float)
                ggx[i] = 1.
                ub = np.inf if ub_reached[i] else 0.
                lb = -np.inf if lb_reached[i] else 0.
                ggs.append(ggx)
                ubs.append(ub)
                lbs.append(lb)
                # print('bounds axis %d -> lambda in (%s, %s)' % (i, str(lb), str(ub)))  # todo: remove
        # check constraints
        for nlc in constraints:
            gx = nlc.fun(state.X)
            ggx = opt.approx_fprime(xk=state.X, f=nlc.fun, epsilon=grad_eps)
            ub = np.inf if (gx > nlc.ub - bounds_tol) else 0.
            lb = -np.inf if (gx < nlc.lb + bounds_tol) else 0.
            if (ub == 0.) and (lb == 0.):
                continue  # constraint not reached
            ggs.append(ggx)
            ubs.append(ub)
            lbs.append(lb)
            # print('constraint -> lambda in (%s, %s)' % (str(lb), str(ub)))  # todo: remove
        # compile and solve F-adjust problem
        # here we have: F_adj = F - ggs.T .matmul. lambda
        # and we optimize: lambda = arcmin ||-F_adj||_2 = arcmin <F_adj|F_adj>
        ggs = np.vstack(ggs)
        ubs = np.array(ubs, dtype=float)
        lbs = np.array(lbs, dtype=float)
        func_F_adj = lambda x: state.F - np.matmul(ggs.T, x)
        problem = lambda x: np.linalg.norm(func_F_adj(x)) ** 2
        problem_grad = lambda x: -2. * np.matmul(ggs, func_F_adj(x))
        result = opt.minimize(
            fun=problem,
            x0=np.zeros(shape=(ggs.shape[0],), dtype=float),
            jac=problem_grad,
            bounds=opt.Bounds(lb=lbs, ub=ubs),
            method='SLSQP', options={'maxiter': 2000},
        )
        if not result.success:
            sys.stdout.write('warning: cannot solve force-adjusted with the constraints.\n')
            return np.copy(state.F)
        s = ('%d steps' % result.nit) if (result.nit != 1) else '1 step'
        sys.stdout.write('force-adjusted: %s in %s\n' % (result.message, s))
        F_adj = func_F_adj(result.x)
        sys.stdout.write('max|F<adjusted>| = %10.7g eV/Ang\n' % max(np.abs(F_adj)))
        return F_adj

    def clean_workspace(self):
        """remove the output files and folders in pwd."""
        os.chdir(pwd)
        for name in ('geometry.in.next_step', 'relaxation_info.dat', 'last.aims.dft.out',
                     'xyz_movie.dat', 'xyz_movie (clean).dat'):
            file = OPJ(pwd, name)
            if os.path.isfile(file):
                os.remove(file)
        for i in range(1000):
            directory = OPJ(pwd, '%d' % i)
            if os.path.isdir(directory):
                shutil.rmtree(directory)
            else:
                break
        os.chdir(pwd)

    def run_aims(self, subfolder, state):
        """
        call AIMS s.c.f. cycle:
        1. run AIMS in subfolder, then return to pwd.
        2. read E and F and fill them inplace in state.
        3. log E and max|F| in corresponding lists.
        """
        sys.stdout.flush()
        # run AIMS
        os.chdir(pwd)
        os.chdir(subfolder)
        os.system(self.run_aims_cmd)
        os.chdir(pwd)
        # read E and F
        state.E, state.free_E, state.F = read_force(OPJ(subfolder, 'aims.dft.out'))
        # log E and max|F|
        self.energies.append(state.E)
        self.free_energies.append(state.free_E)
        self.max_forces.append(max(np.abs(state.F)))
        sys.stdout.write('E = %20.17g eV ,  ' % state.E)
        sys.stdout.write('max|F| = %10.7g eV/Ang\n' % max(np.abs(state.F)))
        sys.stdout.flush()

    def record_state(self, state, is_final=False):
        """make a _State to be self.last_state, write geometry in to "geometry.in.next_step". also
        print trust radius and hess if print_hess is set to True. then, update information in the
        "relaxation_info.dat" file."""
        # save state
        self.last_state = state
        if (self.best_state is None) or (state.E < self.best_state.E):
            self.best_state = state
        # write geometry prediction
        dX = 0. if (state.dX is None) else state.dX  # dX may be undefined
        write_geometry(file_name=OPJ(pwd, 'geometry.in.next_step'),
                       geo_contents=self.geo_contents, coordinate=state.X + dX,
                       trust_radius=state.r if is_final else None,
                       hessian=state.H if is_final else None)
        # write relaxation info
        write_relaxation_info(file_name=OPJ(pwd, 'relaxation_info.dat'),
                              energy=state.E, free_energy=state.free_E,
                              max_force=max(abs(state.F)), accepted=state.accepted,
                              is_final=is_final)

    def check_gliding(self):
        """abort if there are too many recent continuous accepted gliding steps. "gliding" means
        the angle between dX and F is not changing, while energy is not changing, either."""
        # count continuous accepted gliding steps
        count = self.continuous_accepted_gliding_steps(
            angles=self.angles_to_forces,
            energies=self.energies,
            steps_accepted=self.steps_accepted,
        )
        # log if count is big
        if count >= 2:
            s = '1 step has' if (count == 1) else ('%d steps have' % count)
            sys.stdout.write('the most recent %s the same energy and angle<dX,F>\n' % s)
        # abort if meets criteria
        step_number = len(self.energies) - 1
        guaranteed_step_number = 10  # does not abort if this is within the first several steps
        if (count > max_gliding_steps) and (step_number > guaranteed_step_number):
            raise NotConvergedException(
                'reason: all the most recent steps are gliding but not relaxing.\n'
                'this may be caused by the constraints stopping the energy from descending.\n'
                'this is usually the optimal geometry under the constraints.'
            )

    def check_unaccepted(self,):
        """abort if there are too many recent continuous unaccepted steps."""
        # count continuous unaccepted steps
        count = self.continuous_unaccepted_steps(steps_accepted=self.steps_accepted)
        # log if count is big
        if count >= 5:
            s = '1 step is' if (count == 1) else ('%d steps are' % count)
            sys.stdout.write('the most recent %s not accepted\n' % s)
        # abort if meets criteria
        step_number = len(self.energies) - 1
        guaranteed_step_number = 10  # does not abort if this is within the first several steps
        if (count > max_unaccepted_steps) and (step_number > guaranteed_step_number):
            raise NotConvergedException(
                'reason: there are too many steps unaccepted.\n'
                'there are multiple reasons may cause this. you may try to rerun from here.'
            )

    def accept_state(self, state):
        """
        given that this _State is judged to be accepted, this procedure does this:
        1. see if this _State is already converged;
        2. advance the geometry by TRM method;
        3. record the state.
        """
        # update Hessian matrix H
        if state.H is None:  # in the initial step H is already filled in
            self.BFGS_advance(state, self.last_state, assure_posdef=False)
            if np.real(min(np.linalg.eig(state.H)[0])) < epsilon:  # not positive definite
                sys.stdout.write('BFGS update: using Li-Fukushima update to preserve convexness\n')
                self.BFGS_advance(state, self.last_state, assure_posdef=True)
        # judge convergence
        if max(np.abs(state.F)) < self.force_criterion:
            self.not_converged = False
            self.record_state(state)
            sys.stdout.write('converged, quiting relaxation...\n')
            return
        F_adj = self.force_constraint_adjusted(state, bounds=self.bounds, constraints=self.constraints)
        if max(np.abs(F_adj)) < self.force_criterion:
            self.not_converged = False
            self.record_state(state)
            sys.stdout.write('converged considering constraints, quiting relaxation...\n')
            return
        # advance geometry
        self.TRM_advance(state, bounds=self.bounds, constraints=self.constraints)
        self.avoid_atom_collision(state, bounds=self.bounds, constraints=self.constraints)
        self.angles_to_forces.append(angle_between(state.dX, state.F))
        # record state
        self.record_state(state, is_final=False)
        # check if we need to abort
        self.check_unaccepted()
        self.check_gliding()

    def reject_state(self, state):
        """
        given that this _State is judged to be rejected, this procedure does this:
        1. rollback X, E and F to self.last_state;
        2. advance the geometry by TRM method;
        3. record the state.
        """
        # rollback X, E and F
        state.X = self.last_state.X
        state.E = self.last_state.E
        state.F = self.last_state.F
        # # dealing with Hessian
        # eigval, eigvec = np.linalg.eig(self.last_state.H)
        # if np.real(min(eigval)) < epsilon:  # not positive definite
        #     # try to fix Hessian  todo: is this correct?
        #     sys.stdout.write('warning: broken Hessian, trying to fix it\n')
        #     eigval[eigval < 0.] /= 64.
        #     eigval = eigval * np.eye(eigval.shape[0])
        #     state.H = np.matmul(np.matmul(eigvec, eigval), np.linalg.inv(eigvec))
        # else:
        #     # do not advance H otherwise may break its positive definition  todo: correct?
        #     state.H = self.last_feasible_hess
        # do not advance H otherwise may break its positive definition  todo: correct?
        state.H = self.last_state.H
        # determine if to try TRM-advance without Hessian  todo: is this correct?
        non_accepted_count = self.continuous_unaccepted_steps(steps_accepted=self.steps_accepted)
        no_hess = True if (non_accepted_count % 2 == 0) else False
        # advance geometry
        if no_hess:
            state.D = np.random.random(state.X.shape)  # random initial guess
        self.TRM_advance(state, no_hess=no_hess,
                         bounds=self.bounds, constraints=self.constraints)
        self.avoid_atom_collision(state, no_hess=no_hess,
                                  bounds=self.bounds, constraints=self.constraints)
        self.angles_to_forces.append(angle_between(state.dX, state.F))
        # record state
        self.record_state(state, is_final=False)
        # check if we need to abort
        self.check_unaccepted()
        self.check_gliding()

    def initial_iteration(self):
        """the initial step, directly read r and H, always accept step."""
        subfolder = OPJ(pwd, '0')
        sys.stdout.write('\ninitial step\n')
        state = _State()
        time_start = time.time()
        # read control files
        self.force_criteria = read_and_check_control(OPJ(pwd, 'control.in'), modify=False)
        self.geo_contents, state.X = read_geometry(OPJ(pwd, 'geometry.in'))
        if os.path.isfile(OPJ(pwd, 'constraints.in')):
            self.bounds = read_bounds(state.X, OPJ(pwd, 'constraints.in'))
            self.constraints = read_constraints(state.X, OPJ(pwd, 'constraints.in'))
        # read the trust radius if set
        trust_radius = get_and_put_keyword(OPJ(pwd, 'geometry.in'), keyword='trust_radius')
        state.r = float(trust_radius) if trust_radius else default_trust_radius
        # pre-relaxation to satisfy constraints
        self.pre_relaxation(state, bounds=self.bounds, constraints=self.constraints)
        time_prepared = time.time()
        # setup AIMS environment in folder ./0
        os.chdir(pwd)
        os.mkdir(subfolder)
        shutil.copyfile(src=OPJ(pwd, 'control.in'), dst=OPJ(subfolder, 'control.in'))
        read_and_check_control(OPJ(subfolder, 'control.in'), modify=True, max_relaxation_steps=1)
        write_geometry(OPJ(subfolder, 'geometry.in'), geo_contents=self.geo_contents,
                       coordinate=state.X, trust_radius=0.0001)  # provide a very small r
        # run AIMS and read E and F from AIMS results
        self.run_aims(subfolder, state)
        if not os.path.isfile(OPJ(subfolder, 'geometry.in.next_step')):
            raise NotConvergedException(
                'reason: AIMS thinks the input geometry is already converged.\n'
                'is the relaxation really needed? need to modify the input geometry to start.'
            )
        time_scf_done = time.time()
        # get search direction from AIMS results
        if use_aims_prediction:
            _, aims_prediction = read_geometry(OPJ(subfolder, 'geometry.in.next_step'))
            state.D = aims_prediction - state.X
        # directly read H to from AIMS output
        _, state.H = read_hessian(OPJ(subfolder, 'geometry.in.next_step'))
        state.H = init_hess_factor * state.H
        # make H positive definite  todo: is this correct?
        sqrt_epsilon = epsilon ** 0.5  # higher criteria
        eigval, eigvec = np.linalg.eig(state.H)
        if np.real(min(eigval)) < sqrt_epsilon:  # not positive definite
            sys.stdout.write('min(H.eigval) = %8.5g ,  is not positive\n' % np.real(min(eigval)))
            # try to fix Hessian  todo: is this correct?
            sys.stdout.write('warning: broken Hessian, trying to fix it by clipping H.eigval\n')
            eigval[eigval < sqrt_epsilon] = sqrt_epsilon
            eigval = eigval * np.eye(eigval.shape[0])
            state.H = np.matmul(np.matmul(eigvec, eigval), np.linalg.inv(eigvec))
        # the initial step is always accepted
        state.accepted = True
        self.steps_accepted.append(state.accepted)
        # advance
        # self.last_feasible_hess = state.H
        self.accept_state(state)
        # log time consumption
        time_opt_done = time.time()
        sys.stdout.write('time consumption by s.c.f. = %.2f s, by opt. = %.2f s\n' %
                         (time_scf_done - time_prepared,
                          time_opt_done - time_scf_done + time_prepared - time_start))
        sys.stdout.write('\n')
        sys.stdout.flush()

    def iteration(self):
        """regular iteration step."""
        step_number = len(self.energies)
        subfolder = OPJ(pwd, str(step_number))
        sys.stdout.write('\nstep # % 3d\n' % step_number)
        state = _State()
        time_start = time.time()
        # setup AIMS environment in folder ./${step_number}
        os.chdir(pwd)
        os.mkdir(subfolder)
        shutil.copyfile(src=OPJ(pwd, 'control.in'), dst=OPJ(subfolder, 'control.in'))
        read_and_check_control(OPJ(subfolder, 'control.in'), modify=True,
                               max_relaxation_steps=(1 if use_aims_prediction else 0))
        _, state.X = read_geometry(OPJ(pwd, 'geometry.in.next_step'))
        write_geometry(OPJ(subfolder, 'geometry.in'),
                       geo_contents=self.geo_contents, coordinate=state.X)
        # run AIMS and read E and F from AIMS results
        self.run_aims(subfolder, state)
        shutil.copyfile(src=OPJ(subfolder, 'aims.dft.out'), dst=OPJ(pwd, 'last.aims.dft.out'))
        time_scf_done = time.time()
        # get search direction from AIMS results
        if use_aims_prediction:
            if os.path.isfile(OPJ(subfolder, 'geometry.in.next_step')):
                _, aims_prediction = read_geometry(OPJ(subfolder, 'geometry.in.next_step'))
                state.D = aims_prediction - state.X
        # update trust radius r and judge if this step is accepted
        self.TRM_update_radius(state, self.last_state)
        self.steps_accepted.append(state.accepted)
        # advance
        if state.accepted:
            self.last_feasible_hess = self.last_state.H
            self.accept_state(state)
        else:
            sys.stdout.write('step is not accepted, rolling back...\n')
            self.reject_state(state)
        # log time consumption
        time_opt_done = time.time()
        sys.stdout.write('time consumption by s.c.f. = %.2f s, by opt. = %.2f s\n' %
                         (time_scf_done - time_start, time_opt_done - time_scf_done))
        sys.stdout.write('\n')
        sys.stdout.flush()

    def run(self):
        """run relaxation."""
        converged = False  # if the geometry ends up converged
        try:
            self.clean_workspace()
            self.initial_iteration()
            for i in range(max_iterations):
                if os.path.isfile(OPJ(pwd, 'abort_opt')):
                    raise NotConvergedException('manually aborted by "abort_opt".')
                if self.not_converged:
                    self.iteration()
                else:
                    break
            else:
                raise NotConvergedException('the maximum number of iteration exceeded.')
            converged = True
        except NotConvergedException as e:
            # print the reason of abortion
            sys.stdout.write('\n\nrelaxation aborted but not converged:\n')
            sys.stdout.write(str(e))
            sys.stdout.write('\n\n')
        finally:
            if self.best_state is None:
                return  # no need to this wrapping-up
            # restore the best state
            self.record_state(self.best_state, is_final=True)
            while len(self.steps_accepted) < len(self.energies):  # maybe due to some abnormal exit
                self.steps_accepted.append(False)  # now they are equally long
            # append the best state to the lists
            self.energies.append(self.best_state.E)
            self.max_forces.append(max(np.abs(self.best_state.F)))
            self.steps_accepted.append(self.best_state.accepted)
            # write "xyz_movie.dat"
            write_xyz_movie(OPJ(pwd, 'xyz_movie.dat'), workspace=pwd, mask=None)
            # write_xyz_movie(OPJ(pwd, 'xyz_movie (clean).dat'), workspace=pwd,
            #                 mask=self.steps_accepted)
            # log energies
            sys.stdout.write('\n')
            for i in range(len(self.energies)):
                if i == len(self.energies) - 1:
                    sys.stdout.write('finally:  ')
                else:
                    sys.stdout.write('step% 3d:  ' % i)  # starting from 0
                sys.stdout.write('E = %20.17g eV,  ' % self.energies[i])
                sys.stdout.write('max|F| = %8.5g eV/Ang' % self.max_forces[i])
                sys.stdout.write('\n' if self.steps_accepted[i] else ', rejected\n')
            sys.stdout.write('\n')
            # log converged or not
            if converged:
                sys.stdout.write('the final geometry is converged.\nhave a nice day.\n')
            else:
                sys.stdout.write('relaxation is aborted. please see the information above.\n')
# endregion relaxation


if __name__ == '__main__':
    relax = Relaxation()
    relax.run()
