import noumena
import common.default


def migrate_0_4(vars):
    return vars


migrations = [migrate_0_4]


def migrate(vars):
    version = old_version = vars["VERSION"]
    if(old_version != noumena.VERSION):
        for migration in migrations:
            vars = migration(vars)
        vars["VERSION"] = noumena.VERSION
    return vars
