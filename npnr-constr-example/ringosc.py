inv_cell = ctx.cells["inv_LC"]
ctx.bindBel("X1/Y1/lc0", inv_cell, STRENGTH_USER)

osc_net = ctx.nets["clkdiv[0]"]
ctx.bindWire("X1/Y1/lutff_0/out", osc_net, STRENGTH_USER)
ctx.bindPip("X1/Y1/X1.Y1.lutff_0.out.->.X1.Y2.sp4_v_b_5", osc_net, STRENGTH_USER)
ctx.bindPip("X1/Y2/X1.Y2.sp4_v_b_5.->.X1.Y2.sp4_h_r_11", osc_net, STRENGTH_USER)
ctx.bindPip("X5/Y2/X1.Y2.sp4_h_r_11.->.X5.Y2.sp4_v_b_5", osc_net, STRENGTH_USER)
