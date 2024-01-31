// DO NOT EDIT -- auto-generated from riscv-formal/tests/spike/generate.py
#define xlen 32
#define value_xlen value_31_0
#include <stdio.h>
#include <assert.h>
#include "test_add.h"
#include "common.h"
#include "encoding.h"
void test_add(mmu_t mmu, state_t pre_state, insn_t insn)
{
  mmu.optype = 0;
  pre_state.pc = zext_xlen(pre_state.pc) & ~(reg_t)1;
  pre_state.XPR[0] = 0;
  pre_state.XPR[1] = sext_xlen(pre_state.XPR[1]);
  pre_state.XPR[2] = sext_xlen(pre_state.XPR[2]);
  pre_state.XPR[3] = sext_xlen(pre_state.XPR[3]);
  pre_state.XPR[4] = sext_xlen(pre_state.XPR[4]);
  pre_state.XPR[5] = sext_xlen(pre_state.XPR[5]);
  pre_state.XPR[6] = sext_xlen(pre_state.XPR[6]);
  pre_state.XPR[7] = sext_xlen(pre_state.XPR[7]);
  pre_state.XPR[8] = sext_xlen(pre_state.XPR[8]);
  pre_state.XPR[9] = sext_xlen(pre_state.XPR[9]);
  pre_state.XPR[10] = sext_xlen(pre_state.XPR[10]);
  pre_state.XPR[11] = sext_xlen(pre_state.XPR[11]);
  pre_state.XPR[12] = sext_xlen(pre_state.XPR[12]);
  pre_state.XPR[13] = sext_xlen(pre_state.XPR[13]);
  pre_state.XPR[14] = sext_xlen(pre_state.XPR[14]);
  pre_state.XPR[15] = sext_xlen(pre_state.XPR[15]);
  pre_state.XPR[16] = sext_xlen(pre_state.XPR[16]);
  pre_state.XPR[17] = sext_xlen(pre_state.XPR[17]);
  pre_state.XPR[18] = sext_xlen(pre_state.XPR[18]);
  pre_state.XPR[19] = sext_xlen(pre_state.XPR[19]);
  pre_state.XPR[20] = sext_xlen(pre_state.XPR[20]);
  pre_state.XPR[21] = sext_xlen(pre_state.XPR[21]);
  pre_state.XPR[22] = sext_xlen(pre_state.XPR[22]);
  pre_state.XPR[23] = sext_xlen(pre_state.XPR[23]);
  pre_state.XPR[24] = sext_xlen(pre_state.XPR[24]);
  pre_state.XPR[25] = sext_xlen(pre_state.XPR[25]);
  pre_state.XPR[26] = sext_xlen(pre_state.XPR[26]);
  pre_state.XPR[27] = sext_xlen(pre_state.XPR[27]);
  pre_state.XPR[28] = sext_xlen(pre_state.XPR[28]);
  pre_state.XPR[29] = sext_xlen(pre_state.XPR[29]);
  pre_state.XPR[30] = sext_xlen(pre_state.XPR[30]);
  pre_state.XPR[31] = sext_xlen(pre_state.XPR[31]);
  insn.b = sext32(insn.b);
  const reg_t &pc = pre_state.pc;
  reg_t npc = pc + insn.length();
  state_t post_state = pre_state;
  post_state.pc = npc;
  rvfi_insn_add_state_t model = { };
  bool valid = (insn.bits() & MASK_ADD) == MATCH_ADD;
  if (((insn.bits() & 3) != 3) && ((insn.bits() >> 16) != 0)) valid = false;
  model.rvfi_valid.value_0_0 = 1;
  model.rvfi_insn.value_31_0 = insn.bits();
  model.rvfi_pc_rdata.value_xlen = pre_state.pc;
  rvfi_insn_add_init(&model);
  model.rvfi_rs1_rdata.value_xlen = pre_state.XPR[model.spec_rs1_addr.value_4_0];
  model.rvfi_rs2_rdata.value_xlen = pre_state.XPR[model.spec_rs2_addr.value_4_0];
  model.rvfi_mem_rdata.value_xlen = mmu.rdata;
  rvfi_insn_add_eval(&model);
#include "add.h"
  if ((post_state.pc & 1) != 0) valid = false;
#if 0
  printf("int main() {\n"
         "  insn_t insn(%u);\n"
         "  state_t state = { };\n"
         "  mmu_t mmu = { };\n"
         "  state.pc = %u;\n"
         "  state.XPR[%u] = %d;\n"
         "  state.XPR[%u] = %d;\n"
         "  mmu.rdata = %u;\n"
         "  test_add(mmu, state, insn);\n"
         "  return 0;\n"
         "}\n", int(insn.bits()), int(pre_state.pc),
         int(model.spec_rs1_addr.value_4_0), int(pre_state.XPR[model.spec_rs1_addr.value_4_0]),
         int(model.spec_rs2_addr.value_4_0), int(pre_state.XPR[model.spec_rs2_addr.value_4_0]),
         int(mmu.rdata));
  printf("valid: spike=%d riscv-formal=%d\n", int(valid), int(model.spec_valid.value_0_0));
  printf("rs1_addr: riscv-formal=%u\n", int(model.spec_rs1_addr.value_4_0));
  printf("rs2_addr: riscv-formal=%u\n", int(model.spec_rs2_addr.value_4_0));
  printf("rd_addr: riscv-formal=%u\n", int(model.spec_rd_addr.value_4_0));
  printf("rs1_rdata: spike=0x%016llx\n", (long long)pre_state.XPR[model.spec_rs1_addr.value_4_0]);
  printf("rs2_rdata: spike=0x%016llx\n", (long long)pre_state.XPR[model.spec_rs2_addr.value_4_0]);
  printf("rd_wdata: spike=0x%016llx riscv-formal=0x%016llx\n", (long long)post_state.XPR[model.spec_rd_addr.value_4_0], (long long)model.spec_rd_wdata.value_xlen);
#endif
  assert(valid == model.spec_valid.value_0_0);
#if 0
  if (valid) {
    assert(zext_xlen(post_state.pc) == model.spec_pc_wdata.value_xlen);
    assert(post_state.XPR[model.spec_rd_addr.value_4_0] == (reg_t)sext_xlen(model.spec_rd_wdata.value_xlen));
    if (model.spec_rd_addr.value_4_0 != 0) assert(post_state.XPR[0] == pre_state.XPR[0]);
    if (model.spec_rd_addr.value_4_0 != 1) assert(post_state.XPR[1] == pre_state.XPR[1]);
    if (model.spec_rd_addr.value_4_0 != 2) assert(post_state.XPR[2] == pre_state.XPR[2]);
    if (model.spec_rd_addr.value_4_0 != 3) assert(post_state.XPR[3] == pre_state.XPR[3]);
    if (model.spec_rd_addr.value_4_0 != 4) assert(post_state.XPR[4] == pre_state.XPR[4]);
    if (model.spec_rd_addr.value_4_0 != 5) assert(post_state.XPR[5] == pre_state.XPR[5]);
    if (model.spec_rd_addr.value_4_0 != 6) assert(post_state.XPR[6] == pre_state.XPR[6]);
    if (model.spec_rd_addr.value_4_0 != 7) assert(post_state.XPR[7] == pre_state.XPR[7]);
    if (model.spec_rd_addr.value_4_0 != 8) assert(post_state.XPR[8] == pre_state.XPR[8]);
    if (model.spec_rd_addr.value_4_0 != 9) assert(post_state.XPR[9] == pre_state.XPR[9]);
    if (model.spec_rd_addr.value_4_0 != 10) assert(post_state.XPR[10] == pre_state.XPR[10]);
    if (model.spec_rd_addr.value_4_0 != 11) assert(post_state.XPR[11] == pre_state.XPR[11]);
    if (model.spec_rd_addr.value_4_0 != 12) assert(post_state.XPR[12] == pre_state.XPR[12]);
    if (model.spec_rd_addr.value_4_0 != 13) assert(post_state.XPR[13] == pre_state.XPR[13]);
    if (model.spec_rd_addr.value_4_0 != 14) assert(post_state.XPR[14] == pre_state.XPR[14]);
    if (model.spec_rd_addr.value_4_0 != 15) assert(post_state.XPR[15] == pre_state.XPR[15]);
    if (model.spec_rd_addr.value_4_0 != 16) assert(post_state.XPR[16] == pre_state.XPR[16]);
    if (model.spec_rd_addr.value_4_0 != 17) assert(post_state.XPR[17] == pre_state.XPR[17]);
    if (model.spec_rd_addr.value_4_0 != 18) assert(post_state.XPR[18] == pre_state.XPR[18]);
    if (model.spec_rd_addr.value_4_0 != 19) assert(post_state.XPR[19] == pre_state.XPR[19]);
    if (model.spec_rd_addr.value_4_0 != 20) assert(post_state.XPR[20] == pre_state.XPR[20]);
    if (model.spec_rd_addr.value_4_0 != 21) assert(post_state.XPR[21] == pre_state.XPR[21]);
    if (model.spec_rd_addr.value_4_0 != 22) assert(post_state.XPR[22] == pre_state.XPR[22]);
    if (model.spec_rd_addr.value_4_0 != 23) assert(post_state.XPR[23] == pre_state.XPR[23]);
    if (model.spec_rd_addr.value_4_0 != 24) assert(post_state.XPR[24] == pre_state.XPR[24]);
    if (model.spec_rd_addr.value_4_0 != 25) assert(post_state.XPR[25] == pre_state.XPR[25]);
    if (model.spec_rd_addr.value_4_0 != 26) assert(post_state.XPR[26] == pre_state.XPR[26]);
    if (model.spec_rd_addr.value_4_0 != 27) assert(post_state.XPR[27] == pre_state.XPR[27]);
    if (model.spec_rd_addr.value_4_0 != 28) assert(post_state.XPR[28] == pre_state.XPR[28]);
    if (model.spec_rd_addr.value_4_0 != 29) assert(post_state.XPR[29] == pre_state.XPR[29]);
    if (model.spec_rd_addr.value_4_0 != 30) assert(post_state.XPR[30] == pre_state.XPR[30]);
    if (model.spec_rd_addr.value_4_0 != 31) assert(post_state.XPR[31] == pre_state.XPR[31]);
    int8_t model_mem_optype = 100;
    if (model.spec_mem_rmask.value_3_0 ==  0 && model.spec_mem_wmask.value_3_0 ==  0) model_mem_optype =  0;
    if (model.spec_mem_rmask.value_3_0 ==  1 && model.spec_mem_wmask.value_3_0 ==  0) model_mem_optype =  1;
    if (model.spec_mem_rmask.value_3_0 ==  3 && model.spec_mem_wmask.value_3_0 ==  0) model_mem_optype =  2;
    if (model.spec_mem_rmask.value_3_0 == 15 && model.spec_mem_wmask.value_3_0 ==  0) model_mem_optype =  4;
    if (model.spec_mem_rmask.value_3_0 ==  0 && model.spec_mem_wmask.value_3_0 ==  1) model_mem_optype = -1;
    if (model.spec_mem_rmask.value_3_0 ==  0 && model.spec_mem_wmask.value_3_0 ==  3) model_mem_optype = -2;
    if (model.spec_mem_rmask.value_3_0 ==  0 && model.spec_mem_wmask.value_3_0 == 15) model_mem_optype = -4;
    printf("mem_rmask: riscv-formal=%x\n", (int)model.spec_mem_rmask.value_3_0);
    printf("mem_wmask: riscv-formal=%x\n", (int)model.spec_mem_wmask.value_3_0);
    printf("mem_optype: spike=%d riscv-formal=%d\n", (int)mmu.optype, (int)model_mem_optype);
    printf("mem_addr: spike=0x%016llx riscv-formal=0x%016llx\n", (long long)mmu.addr, (long long)sext_xlen(model.spec_mem_addr.value_xlen));
    printf("mem_rdata: spike=0x%016llx riscv-formal=0x%016llx\n", (long long)mmu.rdata, (long long)zext_xlen(model.rvfi_mem_rdata.value_xlen));
    printf("mem_wdata: spike=0x%016llx riscv-formal=0x%016llx\n", (long long)mmu.wdata, (long long)zext_xlen(model.spec_mem_wdata.value_xlen));
    assert(mmu.optype == model_mem_optype);
    if (model_mem_optype)
      assert(zext_xlen(mmu.addr) == zext_xlen(model.spec_mem_addr.value_xlen));
    if (model_mem_optype == -1)
      assert((mmu.wdata & 0xff) == (model.spec_mem_wdata.value_xlen & 0xff));
    if (model_mem_optype == -2)
      assert((mmu.wdata & 0xffff) == (model.spec_mem_wdata.value_xlen & 0xffff));
    if (model_mem_optype == -4)
      assert((mmu.wdata & 0xffffffff) == (model.spec_mem_wdata.value_xlen & 0xffffffff));
    if (model_mem_optype == -8)
      assert(mmu.wdata == model.spec_mem_wdata.value_xlen);
  }
#endif
}
