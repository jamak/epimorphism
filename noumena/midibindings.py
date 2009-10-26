class MidiBindings(object):

    pass




#         if(self.context.midi_controller == "BCF"):
#             self.bindings0 = {(0, 81): [self.zn_set_r_i(0),  "m1(f)", self.zn_get_r_i(0),  "m1_inv(f)", (self.state.zn, 0)],
#                               (0, 82): [self.zn_set_r_i(1),  "m0(f)", self.zn_get_r_i(1),  "m0_inv(f)", (self.state.zn, 1)],
#                               (0, 83): [self.zn_set_r_i(2),  "m1(f)", self.zn_get_r_i(2),  "m1_inv(f)", (self.state.zn, 2)],
#                               (0, 84): [self.zn_set_r_i(3),  "m0(f)", self.zn_get_r_i(3),  "m0_inv(f)", (self.state.zn, 3)],
#                               (0, 85): [self.zn_set_r_i(4),  "m0(f)", self.zn_get_r_i(4),  "m0_inv(f)", (self.state.zn, 4)],
#                               (0, 86): [self.zn_set_r_i(5),  "m0(f)", self.zn_get_r_i(5),  "m0_inv(f)", (self.state.zn, 5)],
#                               (0, 87): [self.zn_set_r_i(6),  "m0(f)", self.zn_get_r_i(6),  "m0_inv(f)", (self.state.zn, 6)],
#                               (0, 88): [self.zn_set_r_i(7),  "m0(f)", self.zn_get_r_i(7),  "m0_inv(f)", (self.state.zn, 7)]}
#             self.bindings0.update(dict([((0, 1 + i), [self.zn_set_th_i(i), "m4(f)", self.zn_get_th_i(i), "m4_inv(f)", (self.state.zn, i)]) for i in xrange(8)]))

#             self.bindings1 = {(0, 81): [self.zn_set_r_i(8),   "m1(f)", self.zn_get_r_i(8),   "m1_inv(f)", (self.state.zn, 8)],
#                               (0, 82): [self.zn_set_r_i(9),   "m0(f)", self.zn_get_r_i(9),   "m0_inv(f)", (self.state.zn, 9)],
#                               (0, 83): [self.zn_set_r_i(10),  "m1(f)", self.zn_get_r_i(10),  "m1_inv(f)", (self.state.zn, 10)],
#                               (0, 84): [self.zn_set_r_i(11),  "m0(f)", self.zn_get_r_i(11),  "m0_inv(f)", (self.state.zn, 11)],
#                               (0, 85): [self.zn_set_r_i(12),  "m0(f)", self.zn_get_r_i(12),  "m0_inv(f)", (self.state.zn, 12)],
#                               (0, 86): [self.zn_set_r_i(13),  "m0(f)", self.zn_get_r_i(13),  "m0_inv(f)", (self.state.zn, 13)],
#                               (0, 87): [self.zn_set_r_i(14),  "m0(f)", self.zn_get_r_i(14),  "m0_inv(f)", (self.state.zn, 14)],
#                               (0, 88): [self.zn_set_r_i(15),  "m0(f)", self.zn_get_r_i(15),  "m0_inv(f)", (self.state.zn, 15)]}
#             self.bindings1.update(dict([((0, 1 + i), [self.zn_set_th_i(8 + i), "m4(f)", self.zn_get_th_i(8 + i), "m4_inv(f)", (self.state.zn, 8 + i)]) for i in xrange(8)]))

#             # create par bindings
#             self.bindings2 = dict([((0, 81 + i), [self.par_set_i(i),      "m0(f)", self.par_get_i(i),      "m0_inv(f)", (self.state.par, i)])      for i in xrange(8)])

#             self.bindings3 = dict([((0, 81 + i), [self.par_set_i(i + 8),  "m0(f)", self.par_get_i(i + 8),  "m0_inv(f)", (self.state.par, i)])      for i in xrange(8)])

#             self.bindings4 = dict([((0, 81 + i), [self.par_set_i(i + 16), "m0(f)", self.par_get_i(i + 16), "m0_inv(f)", (self.state.par, i + 16)]) for i in xrange(8)])

#             self.bindings5 = dict([((0, 81 + i), [self.par_set_i(i + 24), "m0(f)", self.par_get_i(i + 24), "m0_inv(f)", (self.state.par, i + 24)]) for i in xrange(8)])

#             self.bindings6 = dict([((0, 81 + i), [self.par_set_i(i + 32), "m0(f)", self.par_get_i(i + 32), "m0_inv(f)", (self.state.par, i + 32)]) for i in xrange(8)])



#         if(self.context.midi_controller == "UC-33"):
#             self.bindings0 = {(0, 7): [self.zn_set_r_i(0),  "m5(f)", self.zn_get_r_i(0),  "m5_inv(f)", (self.state.zn, 0)],
#                               (1, 7): [self.zn_set_r_i(1),  "m0(f)", self.zn_get_r_i(1),  "m0_inv(f)", (self.state.zn, 1)],
#                               (2, 7): [self.zn_set_r_i(2),  "m5(f)", self.zn_get_r_i(2),  "m5_inv(f)", (self.state.zn, 2)],
#                               (3, 7): [self.zn_set_r_i(3),  "m0(f)", self.zn_get_r_i(3),  "m0_inv(f)", (self.state.zn, 3)],
#                               (4, 7): [self.zn_set_r_i(8),  "m5(f)", self.zn_get_r_i(8),  "m5_inv(f)", (self.state.zn, 8)],
#                               (5, 7): [self.zn_set_r_i(9),  "m0(f)", self.zn_get_r_i(9),  "m0_inv(f)", (self.state.zn, 9)],
#                               (6, 7): [self.zn_set_r_i(10),  "m5(f)", self.zn_get_r_i(10),  "m5_inv(f)", (self.state.zn, 10)],
#                               (7, 7): [self.zn_set_r_i(11),  "m0(f)", self.zn_get_r_i(11),  "m0_inv(f)", (self.state.zn, 11)]}
#             self.bindings0.update({(0, 10): [self.par_set_i(0), "m0_e_e(f)", self.par_get_i(0), "m0_e_e_inv(f)", (self.state.par, 0)]}) # seed_w
#             self.bindings0.update({(1, 10): [self.par_set_i(1), "m0(f)", self.par_get_i(1), "m0_inv(f)", (self.state.par, 1)]}) # color_phi
#             self.bindings0.update({(2, 10): [self.par_set_i(2), "m0(f)", self.par_get_i(2), "m0_inv(f)", (self.state.par, 2)]}) # color_psi
#             self.bindings0.update({(3, 10): [self.par_set_i(3), "m0_e2(f)", self.par_get_i(3), "m0_e2_inv(f)", (self.state.par, 3)]}) # color_a
#             self.bindings0.update({(4, 10): [self.par_set_i(4), "m0_e(f)", self.par_get_i(4), "m0_e_inv(f)", (self.state.par, 4)]}) # color_s
#             self.bindings0.update({(5, 10): [self.par_set_i(13), "m0_e(f)", self.par_get_i(13), "m0_e_inv(f)", (self.state.par, 13)]}) # color_len_sc
#             self.bindings0.update({(6, 10): [self.par_set_i(7), "m0_n_e(f)", self.par_get_i(7), "m0_n_e_inv(f)", (self.state.par, 7)]}) # seed_w_thresh
#             self.bindings0.update({(7, 10): [self.par_set_i(18), "m0(f)", self.par_get_i(18), "m0_inv(f)", (self.state.par, 18)]}) # color_dhue
#             self.bindings0.update({(0, 12): [self.par_set_i(0), "m0_e_e(f)", self.par_get_i(0), "m0_e_e_inv(f)", (self.state.par, 0)]}) # seed_w
#             self.bindings0.update({(1, 12): [self.par_set_i(1), "m0(f)", self.par_get_i(1), "m0_inv(f)", (self.state.par, 1)]}) # color_phi
#             self.bindings0.update({(2, 12): [self.par_set_i(2), "m0(f)", self.par_get_i(2), "m0_inv(f)", (self.state.par, 2)]}) # color_psi
#             self.bindings0.update({(3, 12): [self.par_set_i(3), "m0_e2(f)", self.par_get_i(3), "m0_e2_inv(f)", (self.state.par, 3)]}) # color_a
#             self.bindings0.update({(4, 12): [self.par_set_i(4), "m0_e(f)", self.par_get_i(4), "m0_e_inv(f)", (self.state.par, 4)]}) # color_s
#             self.bindings0.update({(5, 12): [self.par_set_i(13), "m0_e(f)", self.par_get_i(13), "m0_e_inv(f)", (self.state.par, 13)]}) # color_len_sc
#             self.bindings0.update({(6, 12): [self.par_set_i(7), "m0_n_e(f)", self.par_get_i(7), "m0_n_e_inv(f)", (self.state.par, 7)]}) # seed_w_thresh
#             self.bindings0.update({(7, 12): [self.par_set_i(18), "m0(f)", self.par_get_i(18), "m0_inv(f)", (self.state.par, 18)]}) # color_dhue
#             self.bindings0.update({(0, 13): [self.par_set_i(0), "m0_e_e(f)", self.par_get_i(0), "m0_e_e_inv(f)", (self.state.par, 0)]}) # seed_w
#             self.bindings0.update({(1, 13): [self.par_set_i(1), "m0(f)", self.par_get_i(1), "m0_inv(f)", (self.state.par, 1)]}) # color_phi
#             self.bindings0.update({(2, 13): [self.par_set_i(2), "m0(f)", self.par_get_i(2), "m0_inv(f)", (self.state.par, 2)]}) # color_psi
#             self.bindings0.update({(3, 13): [self.par_set_i(3), "m0_e2(f)", self.par_get_i(3), "m0_e2_inv(f)", (self.state.par, 3)]}) # color_a
#             self.bindings0.update({(4, 13): [self.par_set_i(4), "m0_e(f)", self.par_get_i(4), "m0_e_inv(f)", (self.state.par, 4)]}) # color_s
#             self.bindings0.update({(5, 13): [self.par_set_i(13), "m0_e(f)", self.par_get_i(13), "m0_e_inv(f)", (self.state.par, 13)]}) # color_len_sc
#             self.bindings0.update({(6, 13): [self.par_set_i(7), "m0_n_e(f)", self.par_get_i(7), "m0_n_e_inv(f)", (self.state.par, 7)]}) # seed_w_thresh
#            self.bindings0.update({(7, 13): [self.par_set_i(18), "m0(f)", self.par_get_i(18), "m0_inv(f)", (self.state.par, 18)]}) # color_dhue
