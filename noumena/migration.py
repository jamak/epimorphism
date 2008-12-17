import noumena
import common.default


# test migration.  remove whenever we have a real migration
def migrate_0_4(vars):
    return vars


# list of all migrations, in order
migrations = [migrate_0_4]


def migrate(vars):

    # test if update necessary
    old_version = vars["VERSION"]
    if(old_version != noumena.VERSION):

        # update
        for migration in migrations:
            vars = migration(vars)

        # update VERSION
        vars["VERSION"] = noumena.VERSION

    return vars
