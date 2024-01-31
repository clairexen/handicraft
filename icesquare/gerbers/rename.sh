#!/bin/bash
set -ex
mv icesquare-B.Cu.gbr       ICESQUARE.GBL
mv icesquare-F.Cu.gbr       ICESQUARE.GTL
mv icesquare-F.Paste.gbr    ICESQUARE.GTP
mv icesquare-F.SilkS.gbr    ICESQUARE.GTO
mv icesquare-B.Mask.gbr     ICESQUARE.GBS
mv icesquare-F.Mask.gbr     ICESQUARE.GTS
mv icesquare-Edge.Cuts.gbr  ICESQUARE.GKO
mv icesquare.drl            ICESQUARE.TXT
rm -f icesquare-NPTH.drl
