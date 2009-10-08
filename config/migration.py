import noumena
import common.default

import copy

from phenom.datamanager import *

def relayout_pars(vars, new_names, new_defaults):
    old_par = copy.copy(vars["par"])

    for i in xrange(len(new_names)):
        name = new_names[i]
        if(new_defaults.has_key(name)):
            vars["par"][i] = new_defaults[name]
        else:
            idx = vars["par_names"].index(name)
            vars["par"][i] = old_par[idx]

    vars["par_names"] = new_names


def migrate_0_93(vars):
    print "executing migration 0.93"

    if(vars["T"] == "zn[0] * (zn[2] * z + zn[3]) + zn[1]"):
        vars["T"] = "zn[0] * ((zn[2] * z + zn[3])) + zn[1]"

    if(vars["T_SEED"] == "zn[8] * (zn[10] * z + zn[11]) + zn[9]"):
        vars["T_SEED"] = "zn[8] * ((zn[10] * z + zn[11])) + zn[9]"

    if(vars.has_key("component_vals")):
        del vars["component_vals"]

    if(vars.has_key("component_idx")):
        del vars["component_idx"]

    #print vars


def migrate_0_916(vars):
    print "executing migration 0.916"

    vars["par"][vars["par_names"].index("_COLOR_PHI")] *= 2.0
    vars["par"][vars["par_names"].index("_COLOR_PSI")] *= 2.0


def migrate_0_91(vars):
    print "executing migration 0.91"

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


# dict of all migrations
all_migrations = {0.91: migrate_0_91, 0.916: migrate_0_916, 0.93: migrate_0_93}


def migrate(vars):

    # test if update necessary
    old_version = vars["VERSION"]
    if(old_version < noumena.VERSION):


        # get necessary migrations
        migrations = all_migrations.keys()
        migrations = [version for version in migrations if version > old_version]
        migrations.sort()

        # run migrations
        for version in migrations:
            all_migrations[version](vars)

        # update VERSION
        vars["VERSION"] = noumena.VERSION

    return vars
