#  EPIMORPHISM - v3.0b
#
#  The project is structured into 4 branches:
#    noumena - the parametric engine which creates & displays
#    phenom  - the control system for noumena
#    aer     - the banks of data/libraries phenom uses to control noumena
#    common  - common functionality across packages & utility libraries
#
#  The project uses 3 data structures for configuration - state/profile/context
#    see common/state.py for reference
#
#
#  COMMAND LINE ARGUMENTS:
#
#  ex. ./epimorph.py ~state:state2 %T:"z + zn[0] * 4.0" ~midi:4==4
#    will load the program with the state: state2.est, profile: box1.prf, context: default.ctx
#    it will also set state.T = "z + zn[0] * 4.0" and context.midi = eval(4==4) = True
#
