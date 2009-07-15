import noumena
import common.default

import copy

from phenom.datamanager import *

def relayout_pars(vars, new_names, new_defaults):
    old_par = copy.copy(vars["par"])

    vars["par_defaults"].update(new_defaults)

    for i in xrange(len(new_names)):
        name = new_names[i]
        try:
            idx = vars["par_names"].index(name)
            vars["par"][i] = old_par[idx]
        except:
            vars["par"][i] = vars["par_defaults"][name]

    vars["par_names"] = new_names

def migrate_0_92(vars):

    datamanager = DataManager()

    vars["component_vals"] = [0 for i in xrange(10)]

    return vars

def migrate_0_916(vars):

    vars["par"][vars["par_names"].index("_COLOR_PHI")] *= 2.0
    vars["par"][vars["par_names"].index("_COLOR_PSI")] *= 2.0

    return vars

def migrate_0_91(vars):

    new_names = [
        '_SEED_W',
        '_COLOR_PHI',
        '_COLOR_PSI',
        '_COLOR_A',
        '_COLOR_S',
        '_COLOR_V',
        '_SEED_W_BASE',
        '_SEED_W_THRESH',
        '_SEED_C_TH_EFF',
        '_COLOR_TH_EFF',
        '_COLOR_SPEED_TH',
        '_MOD_SPEED_COLOR',
        '_GLOBAL_SPEED',
        '_COLOR_LEN_SC',
        '_SEED_CIRCLE_R',
        '_SEED_GRID_N',
        '_HSLS_RESET_Z',
        '_CULL_DEPTH',
        '_COLOR_DHUE'
        ]

    new_defaults = {
        '_COLOR_V' : 1.0,
        '_HSLS_RESET_Z' : 0.5,
        }

    relayout_pars(vars, new_names, new_defaults)

    vars["RESET"] = "reset_hsls(x, y)"

    vars["zn"] += [0j for i in xrange(6)]

    return vars


# dict of all migrations
migrations = {0.91 : migrate_0_91, 0.916 : migrate_0_916, 0.92 : migrate_0_92}


def migrate(vars):

    # test if update necessary
    old_version = vars["VERSION"]
    if(old_version != noumena.VERSION):

        # update

        versions = migrations.keys()
        versions.sort()
        versions = [version for version in versions if version > old_version]

        for version in versions:
            vars = migrations[version](vars)

        # update VERSION
        vars["VERSION"] = noumena.VERSION

    return vars
