def extra_vlog_files(basedir, mod):
    vlog = set()
    vlog.add("%s/mod_ioshim/ioshim_alu.v" % basedir)
    vlog.add("%s/mod_ioshim/ioshim_cpu.v" % basedir)
    vlog.add("%s/mod_ioshim/ioshim_mem.v" % basedir)
    vlog.add("%s/mod_ioshim/ioshim_regs.v" % basedir)
    vlog.add("%s/mod_ioshim/ioshim_gpio.v" % basedir)
    return vlog
