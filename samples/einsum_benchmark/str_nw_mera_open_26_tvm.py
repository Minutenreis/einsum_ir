import tvm
import tvm.auto_scheduler
import tvm_helper
import argparse
##
# Note: This script was automatically generated by tvm_from_tree.py.
#
# input_string: [[[[[8,9,44]->[9,8,44]],[[17,9,11]->[11,17,9]]->[11,17,8,44]],[7,8]->[11,17,7,44]],[[6,7,43]->[6,43,7]]->[11,6,17,43,44]],[[[[[[[[15,17,31]->[15,31,17]],[[30,31,32,33]->[32,30,33,31]]->[32,30,15,33,17]],[[[[[13,15,19]->[19,13,15]],[[[12,13],[[12,14,18]->[14,18,12]]->[14,18,13]],[[18,19,20,21]->[20,21,19,18]]->[14,20,21,19,13]]->[14,20,21,15]],[[21,27,30]->[27,30,21]]->[14,20,27,30,15]],[[[26,27,28,29]->[28,29,26,27]],[[20,23,26]->[23,20,26]]->[23,28,29,20,27]]->[14,23,28,29,30,15]]->[14,23,28,29,32,33,17]],[[[29,32,1,2]->[1,2,29,32]],[[2,3,41]->[41,3,2]]->[41,1,3,29,32]]->[41,14,23,1,28,3,33,17]],[[33,3,4]->[4,3,33]]->[41,14,23,1,28,4,17]],[[[[[25,28,37,0]->[25,37,0,28]],[[0,1,40]->[40,1,0]]->[40,25,37,1,28]],[[36,37,39]->[39,36,37]]->[39,40,36,25,1,28]],[[[[[[22,23,24,25]->[23,22,24,25]],[[24,35,36]->[35,36,24]]->[23,35,22,36,25]],[14,16,22]->[14,23,35,16,36,25]],[[16,34,10]->[10,34,16]]->[10,14,23,35,34,36,25]],[[34,35,38]->[38,35,34]]->[10,38,14,23,36,25]]->[10,38,14,23,39,40,1,28]]->[10,38,39,40,41,4,17]],[[[4,5,42]->[42,5,4]],[[5,6]->[6,5]]->[42,6,4]]->[10,38,39,40,41,42,6,17]]->[10,11,38,39,40,41,42,43,44]
##
@tvm.auto_scheduler.register_workload
def einsum_tree( dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dim_9, dim_10, dim_11, dim_12, dim_13, dim_14, dim_15, dim_16, dim_17, dim_18, dim_19, dim_20, dim_21, dim_22, dim_23, dim_24, dim_25, dim_26, dim_27, dim_28, dim_29, dim_30, dim_31, dim_32, dim_33, dim_34, dim_35, dim_36, dim_37, dim_38, dim_39, dim_40, dim_41, dim_42, dim_43, dim_44, dtype):
  tensor_4_5_42 = tvm.te.placeholder((dim_4, dim_5, dim_42), name='tensor_4_5_42', dtype=dtype)
  tensor_5_6 = tvm.te.placeholder((dim_5, dim_6), name='tensor_5_6', dtype=dtype)
  tensor_22_23_24_25 = tvm.te.placeholder((dim_22, dim_23, dim_24, dim_25), name='tensor_22_23_24_25', dtype=dtype)
  tensor_24_35_36 = tvm.te.placeholder((dim_24, dim_35, dim_36), name='tensor_24_35_36', dtype=dtype)
  tensor_25_28_37_0 = tvm.te.placeholder((dim_25, dim_28, dim_37, dim_0), name='tensor_25_28_37_0', dtype=dtype)
  tensor_0_1_40 = tvm.te.placeholder((dim_0, dim_1, dim_40), name='tensor_0_1_40', dtype=dtype)
  tensor_29_32_1_2 = tvm.te.placeholder((dim_29, dim_32, dim_1, dim_2), name='tensor_29_32_1_2', dtype=dtype)
  tensor_2_3_41 = tvm.te.placeholder((dim_2, dim_3, dim_41), name='tensor_2_3_41', dtype=dtype)
  tensor_26_27_28_29 = tvm.te.placeholder((dim_26, dim_27, dim_28, dim_29), name='tensor_26_27_28_29', dtype=dtype)
  tensor_20_23_26 = tvm.te.placeholder((dim_20, dim_23, dim_26), name='tensor_20_23_26', dtype=dtype)
  tensor_12_13 = tvm.te.placeholder((dim_12, dim_13), name='tensor_12_13', dtype=dtype)
  tensor_12_14_18 = tvm.te.placeholder((dim_12, dim_14, dim_18), name='tensor_12_14_18', dtype=dtype)
  tensor_15_17_31 = tvm.te.placeholder((dim_15, dim_17, dim_31), name='tensor_15_17_31', dtype=dtype)
  tensor_30_31_32_33 = tvm.te.placeholder((dim_30, dim_31, dim_32, dim_33), name='tensor_30_31_32_33', dtype=dtype)
  tensor_8_9_44 = tvm.te.placeholder((dim_8, dim_9, dim_44), name='tensor_8_9_44', dtype=dtype)
  tensor_17_9_11 = tvm.te.placeholder((dim_17, dim_9, dim_11), name='tensor_17_9_11', dtype=dtype)
  tensor_34_35_38 = tvm.te.placeholder((dim_34, dim_35, dim_38), name='tensor_34_35_38', dtype=dtype)
  tensor_16_34_10 = tvm.te.placeholder((dim_16, dim_34, dim_10), name='tensor_16_34_10', dtype=dtype)
  tensor_14_16_22 = tvm.te.placeholder((dim_14, dim_16, dim_22), name='tensor_14_16_22', dtype=dtype)
  tensor_36_37_39 = tvm.te.placeholder((dim_36, dim_37, dim_39), name='tensor_36_37_39', dtype=dtype)
  tensor_33_3_4 = tvm.te.placeholder((dim_33, dim_3, dim_4), name='tensor_33_3_4', dtype=dtype)
  tensor_21_27_30 = tvm.te.placeholder((dim_21, dim_27, dim_30), name='tensor_21_27_30', dtype=dtype)
  tensor_18_19_20_21 = tvm.te.placeholder((dim_18, dim_19, dim_20, dim_21), name='tensor_18_19_20_21', dtype=dtype)
  tensor_13_15_19 = tvm.te.placeholder((dim_13, dim_15, dim_19), name='tensor_13_15_19', dtype=dtype)
  tensor_6_7_43 = tvm.te.placeholder((dim_6, dim_7, dim_43), name='tensor_6_7_43', dtype=dtype)
  tensor_7_8 = tvm.te.placeholder((dim_7, dim_8), name='tensor_7_8', dtype=dtype)

  tmp_12 = tvm.te.reduce_axis((0, dim_12), name='tmp_12')
  tmp_14 = tvm.te.reduce_axis((0, dim_14), name='tmp_14')
  tmp_29 = tvm.te.reduce_axis((0, dim_29), name='tmp_29')
  tmp_22 = tvm.te.reduce_axis((0, dim_22), name='tmp_22')
  tmp_37 = tvm.te.reduce_axis((0, dim_37), name='tmp_37')
  tmp_8 = tvm.te.reduce_axis((0, dim_8), name='tmp_8')
  tmp_3 = tvm.te.reduce_axis((0, dim_3), name='tmp_3')
  tmp_34 = tvm.te.reduce_axis((0, dim_34), name='tmp_34')
  tmp_20 = tvm.te.reduce_axis((0, dim_20), name='tmp_20')
  tmp_32 = tvm.te.reduce_axis((0, dim_32), name='tmp_32')
  tmp_4 = tvm.te.reduce_axis((0, dim_4), name='tmp_4')
  tmp_1 = tvm.te.reduce_axis((0, dim_1), name='tmp_1')
  tmp_21 = tvm.te.reduce_axis((0, dim_21), name='tmp_21')
  tmp_6 = tvm.te.reduce_axis((0, dim_6), name='tmp_6')
  tmp_28 = tvm.te.reduce_axis((0, dim_28), name='tmp_28')
  tmp_9 = tvm.te.reduce_axis((0, dim_9), name='tmp_9')
  tmp_27 = tvm.te.reduce_axis((0, dim_27), name='tmp_27')
  tmp_36 = tvm.te.reduce_axis((0, dim_36), name='tmp_36')
  tmp_26 = tvm.te.reduce_axis((0, dim_26), name='tmp_26')
  tmp_7 = tvm.te.reduce_axis((0, dim_7), name='tmp_7')
  tmp_30 = tvm.te.reduce_axis((0, dim_30), name='tmp_30')
  tmp_25 = tvm.te.reduce_axis((0, dim_25), name='tmp_25')
  tmp_13 = tvm.te.reduce_axis((0, dim_13), name='tmp_13')
  tmp_17 = tvm.te.reduce_axis((0, dim_17), name='tmp_17')
  tmp_35 = tvm.te.reduce_axis((0, dim_35), name='tmp_35')
  tmp_5 = tvm.te.reduce_axis((0, dim_5), name='tmp_5')
  tmp_18 = tvm.te.reduce_axis((0, dim_18), name='tmp_18')
  tmp_16 = tvm.te.reduce_axis((0, dim_16), name='tmp_16')
  tmp_33 = tvm.te.reduce_axis((0, dim_33), name='tmp_33')
  tmp_2 = tvm.te.reduce_axis((0, dim_2), name='tmp_2')
  tmp_15 = tvm.te.reduce_axis((0, dim_15), name='tmp_15')
  tmp_19 = tvm.te.reduce_axis((0, dim_19), name='tmp_19')
  tmp_0 = tvm.te.reduce_axis((0, dim_0), name='tmp_0')
  tmp_24 = tvm.te.reduce_axis((0, dim_24), name='tmp_24')
  tmp_31 = tvm.te.reduce_axis((0, dim_31), name='tmp_31')
  tmp_23 = tvm.te.reduce_axis((0, dim_23), name='tmp_23')

  tensor_42_6_4 = tvm.te.compute( (dim_42, dim_6, dim_4), lambda tmp_42, tmp_6, tmp_4: tvm.te.sum( tensor_4_5_42[ tmp_4, tmp_5, tmp_42 ] * tensor_5_6[ tmp_5, tmp_6 ] , axis=[ tmp_5 ]), name='tensor_42_6_4' )
  tensor_23_35_22_36_25 = tvm.te.compute( (dim_23, dim_35, dim_22, dim_36, dim_25), lambda tmp_23, tmp_35, tmp_22, tmp_36, tmp_25: tvm.te.sum( tensor_22_23_24_25[ tmp_22, tmp_23, tmp_24, tmp_25 ] * tensor_24_35_36[ tmp_24, tmp_35, tmp_36 ] , axis=[ tmp_24 ]), name='tensor_23_35_22_36_25' )
  tensor_14_23_35_16_36_25 = tvm.te.compute( (dim_14, dim_23, dim_35, dim_16, dim_36, dim_25), lambda tmp_14, tmp_23, tmp_35, tmp_16, tmp_36, tmp_25: tvm.te.sum( tensor_23_35_22_36_25[ tmp_23, tmp_35, tmp_22, tmp_36, tmp_25 ] * tensor_14_16_22[ tmp_14, tmp_16, tmp_22 ] , axis=[ tmp_22 ]), name='tensor_14_23_35_16_36_25' )
  tensor_10_14_23_35_34_36_25 = tvm.te.compute( (dim_10, dim_14, dim_23, dim_35, dim_34, dim_36, dim_25), lambda tmp_10, tmp_14, tmp_23, tmp_35, tmp_34, tmp_36, tmp_25: tvm.te.sum( tensor_14_23_35_16_36_25[ tmp_14, tmp_23, tmp_35, tmp_16, tmp_36, tmp_25 ] * tensor_16_34_10[ tmp_16, tmp_34, tmp_10 ] , axis=[ tmp_16 ]), name='tensor_10_14_23_35_34_36_25' )
  tensor_10_38_14_23_36_25 = tvm.te.compute( (dim_10, dim_38, dim_14, dim_23, dim_36, dim_25), lambda tmp_10, tmp_38, tmp_14, tmp_23, tmp_36, tmp_25: tvm.te.sum( tensor_10_14_23_35_34_36_25[ tmp_10, tmp_14, tmp_23, tmp_35, tmp_34, tmp_36, tmp_25 ] * tensor_34_35_38[ tmp_34, tmp_35, tmp_38 ] , axis=[ tmp_35, tmp_34 ]), name='tensor_10_38_14_23_36_25' )
  tensor_40_25_37_1_28 = tvm.te.compute( (dim_40, dim_25, dim_37, dim_1, dim_28), lambda tmp_40, tmp_25, tmp_37, tmp_1, tmp_28: tvm.te.sum( tensor_25_28_37_0[ tmp_25, tmp_28, tmp_37, tmp_0 ] * tensor_0_1_40[ tmp_0, tmp_1, tmp_40 ] , axis=[ tmp_0 ]), name='tensor_40_25_37_1_28' )
  tensor_39_40_36_25_1_28 = tvm.te.compute( (dim_39, dim_40, dim_36, dim_25, dim_1, dim_28), lambda tmp_39, tmp_40, tmp_36, tmp_25, tmp_1, tmp_28: tvm.te.sum( tensor_40_25_37_1_28[ tmp_40, tmp_25, tmp_37, tmp_1, tmp_28 ] * tensor_36_37_39[ tmp_36, tmp_37, tmp_39 ] , axis=[ tmp_37 ]), name='tensor_39_40_36_25_1_28' )
  tensor_10_38_14_23_39_40_1_28 = tvm.te.compute( (dim_10, dim_38, dim_14, dim_23, dim_39, dim_40, dim_1, dim_28), lambda tmp_10, tmp_38, tmp_14, tmp_23, tmp_39, tmp_40, tmp_1, tmp_28: tvm.te.sum( tensor_39_40_36_25_1_28[ tmp_39, tmp_40, tmp_36, tmp_25, tmp_1, tmp_28 ] * tensor_10_38_14_23_36_25[ tmp_10, tmp_38, tmp_14, tmp_23, tmp_36, tmp_25 ] , axis=[ tmp_36, tmp_25 ]), name='tensor_10_38_14_23_39_40_1_28' )
  tensor_41_1_3_29_32 = tvm.te.compute( (dim_41, dim_1, dim_3, dim_29, dim_32), lambda tmp_41, tmp_1, tmp_3, tmp_29, tmp_32: tvm.te.sum( tensor_29_32_1_2[ tmp_29, tmp_32, tmp_1, tmp_2 ] * tensor_2_3_41[ tmp_2, tmp_3, tmp_41 ] , axis=[ tmp_2 ]), name='tensor_41_1_3_29_32' )
  tensor_23_28_29_20_27 = tvm.te.compute( (dim_23, dim_28, dim_29, dim_20, dim_27), lambda tmp_23, tmp_28, tmp_29, tmp_20, tmp_27: tvm.te.sum( tensor_26_27_28_29[ tmp_26, tmp_27, tmp_28, tmp_29 ] * tensor_20_23_26[ tmp_20, tmp_23, tmp_26 ] , axis=[ tmp_26 ]), name='tensor_23_28_29_20_27' )
  tensor_14_18_13 = tvm.te.compute( (dim_14, dim_18, dim_13), lambda tmp_14, tmp_18, tmp_13: tvm.te.sum( tensor_12_13[ tmp_12, tmp_13 ] * tensor_12_14_18[ tmp_12, tmp_14, tmp_18 ] , axis=[ tmp_12 ]), name='tensor_14_18_13' )
  tensor_14_20_21_19_13 = tvm.te.compute( (dim_14, dim_20, dim_21, dim_19, dim_13), lambda tmp_14, tmp_20, tmp_21, tmp_19, tmp_13: tvm.te.sum( tensor_14_18_13[ tmp_14, tmp_18, tmp_13 ] * tensor_18_19_20_21[ tmp_18, tmp_19, tmp_20, tmp_21 ] , axis=[ tmp_18 ]), name='tensor_14_20_21_19_13' )
  tensor_14_20_21_15 = tvm.te.compute( (dim_14, dim_20, dim_21, dim_15), lambda tmp_14, tmp_20, tmp_21, tmp_15: tvm.te.sum( tensor_13_15_19[ tmp_13, tmp_15, tmp_19 ] * tensor_14_20_21_19_13[ tmp_14, tmp_20, tmp_21, tmp_19, tmp_13 ] , axis=[ tmp_19, tmp_13 ]), name='tensor_14_20_21_15' )
  tensor_14_20_27_30_15 = tvm.te.compute( (dim_14, dim_20, dim_27, dim_30, dim_15), lambda tmp_14, tmp_20, tmp_27, tmp_30, tmp_15: tvm.te.sum( tensor_14_20_21_15[ tmp_14, tmp_20, tmp_21, tmp_15 ] * tensor_21_27_30[ tmp_21, tmp_27, tmp_30 ] , axis=[ tmp_21 ]), name='tensor_14_20_27_30_15' )
  tensor_14_23_28_29_30_15 = tvm.te.compute( (dim_14, dim_23, dim_28, dim_29, dim_30, dim_15), lambda tmp_14, tmp_23, tmp_28, tmp_29, tmp_30, tmp_15: tvm.te.sum( tensor_14_20_27_30_15[ tmp_14, tmp_20, tmp_27, tmp_30, tmp_15 ] * tensor_23_28_29_20_27[ tmp_23, tmp_28, tmp_29, tmp_20, tmp_27 ] , axis=[ tmp_27, tmp_20 ]), name='tensor_14_23_28_29_30_15' )
  tensor_32_30_15_33_17 = tvm.te.compute( (dim_32, dim_30, dim_15, dim_33, dim_17), lambda tmp_32, tmp_30, tmp_15, tmp_33, tmp_17: tvm.te.sum( tensor_15_17_31[ tmp_15, tmp_17, tmp_31 ] * tensor_30_31_32_33[ tmp_30, tmp_31, tmp_32, tmp_33 ] , axis=[ tmp_31 ]), name='tensor_32_30_15_33_17' )
  tensor_14_23_28_29_32_33_17 = tvm.te.compute( (dim_14, dim_23, dim_28, dim_29, dim_32, dim_33, dim_17), lambda tmp_14, tmp_23, tmp_28, tmp_29, tmp_32, tmp_33, tmp_17: tvm.te.sum( tensor_32_30_15_33_17[ tmp_32, tmp_30, tmp_15, tmp_33, tmp_17 ] * tensor_14_23_28_29_30_15[ tmp_14, tmp_23, tmp_28, tmp_29, tmp_30, tmp_15 ] , axis=[ tmp_30, tmp_15 ]), name='tensor_14_23_28_29_32_33_17' )
  tensor_41_14_23_1_28_3_33_17 = tvm.te.compute( (dim_41, dim_14, dim_23, dim_1, dim_28, dim_3, dim_33, dim_17), lambda tmp_41, tmp_14, tmp_23, tmp_1, tmp_28, tmp_3, tmp_33, tmp_17: tvm.te.sum( tensor_14_23_28_29_32_33_17[ tmp_14, tmp_23, tmp_28, tmp_29, tmp_32, tmp_33, tmp_17 ] * tensor_41_1_3_29_32[ tmp_41, tmp_1, tmp_3, tmp_29, tmp_32 ] , axis=[ tmp_29, tmp_32 ]), name='tensor_41_14_23_1_28_3_33_17' )
  tensor_41_14_23_1_28_4_17 = tvm.te.compute( (dim_41, dim_14, dim_23, dim_1, dim_28, dim_4, dim_17), lambda tmp_41, tmp_14, tmp_23, tmp_1, tmp_28, tmp_4, tmp_17: tvm.te.sum( tensor_41_14_23_1_28_3_33_17[ tmp_41, tmp_14, tmp_23, tmp_1, tmp_28, tmp_3, tmp_33, tmp_17 ] * tensor_33_3_4[ tmp_33, tmp_3, tmp_4 ] , axis=[ tmp_3, tmp_33 ]), name='tensor_41_14_23_1_28_4_17' )
  tensor_10_38_39_40_41_4_17 = tvm.te.compute( (dim_10, dim_38, dim_39, dim_40, dim_41, dim_4, dim_17), lambda tmp_10, tmp_38, tmp_39, tmp_40, tmp_41, tmp_4, tmp_17: tvm.te.sum( tensor_41_14_23_1_28_4_17[ tmp_41, tmp_14, tmp_23, tmp_1, tmp_28, tmp_4, tmp_17 ] * tensor_10_38_14_23_39_40_1_28[ tmp_10, tmp_38, tmp_14, tmp_23, tmp_39, tmp_40, tmp_1, tmp_28 ] , axis=[ tmp_23, tmp_28, tmp_1, tmp_14 ]), name='tensor_10_38_39_40_41_4_17' )
  tensor_10_38_39_40_41_42_6_17 = tvm.te.compute( (dim_10, dim_38, dim_39, dim_40, dim_41, dim_42, dim_6, dim_17), lambda tmp_10, tmp_38, tmp_39, tmp_40, tmp_41, tmp_42, tmp_6, tmp_17: tvm.te.sum( tensor_10_38_39_40_41_4_17[ tmp_10, tmp_38, tmp_39, tmp_40, tmp_41, tmp_4, tmp_17 ] * tensor_42_6_4[ tmp_42, tmp_6, tmp_4 ] , axis=[ tmp_4 ]), name='tensor_10_38_39_40_41_42_6_17' )
  tensor_11_17_8_44 = tvm.te.compute( (dim_11, dim_17, dim_8, dim_44), lambda tmp_11, tmp_17, tmp_8, tmp_44: tvm.te.sum( tensor_8_9_44[ tmp_8, tmp_9, tmp_44 ] * tensor_17_9_11[ tmp_17, tmp_9, tmp_11 ] , axis=[ tmp_9 ]), name='tensor_11_17_8_44' )
  tensor_11_17_7_44 = tvm.te.compute( (dim_11, dim_17, dim_7, dim_44), lambda tmp_11, tmp_17, tmp_7, tmp_44: tvm.te.sum( tensor_11_17_8_44[ tmp_11, tmp_17, tmp_8, tmp_44 ] * tensor_7_8[ tmp_7, tmp_8 ] , axis=[ tmp_8 ]), name='tensor_11_17_7_44' )
  tensor_11_6_17_43_44 = tvm.te.compute( (dim_11, dim_6, dim_17, dim_43, dim_44), lambda tmp_11, tmp_6, tmp_17, tmp_43, tmp_44: tvm.te.sum( tensor_11_17_7_44[ tmp_11, tmp_17, tmp_7, tmp_44 ] * tensor_6_7_43[ tmp_6, tmp_7, tmp_43 ] , axis=[ tmp_7 ]), name='tensor_11_6_17_43_44' )
  tensor_10_11_38_39_40_41_42_43_44 = tvm.te.compute( (dim_10, dim_11, dim_38, dim_39, dim_40, dim_41, dim_42, dim_43, dim_44), lambda tmp_10, tmp_11, tmp_38, tmp_39, tmp_40, tmp_41, tmp_42, tmp_43, tmp_44: tvm.te.sum( tensor_11_6_17_43_44[ tmp_11, tmp_6, tmp_17, tmp_43, tmp_44 ] * tensor_10_38_39_40_41_42_6_17[ tmp_10, tmp_38, tmp_39, tmp_40, tmp_41, tmp_42, tmp_6, tmp_17 ] , axis=[ tmp_17, tmp_6 ]), name='tensor_10_11_38_39_40_41_42_43_44' )

  return [ tensor_4_5_42, tensor_5_6, tensor_22_23_24_25, tensor_24_35_36, tensor_25_28_37_0, tensor_0_1_40, tensor_29_32_1_2, tensor_2_3_41, tensor_26_27_28_29, tensor_20_23_26, tensor_12_13, tensor_12_14_18, tensor_15_17_31, tensor_30_31_32_33, tensor_8_9_44, tensor_17_9_11, tensor_34_35_38, tensor_16_34_10, tensor_14_16_22, tensor_36_37_39, tensor_33_3_4, tensor_21_27_30, tensor_18_19_20_21, tensor_13_15_19, tensor_6_7_43, tensor_7_8, tensor_10_11_38_39_40_41_42_43_44 ]

if __name__=="__main__":
  args = tvm_helper.parse_args()

  target = tvm.target.Target( tvm_helper.cpu_to_llvm( args.cpu ) )
  hardware_params = tvm.auto_scheduler.HardwareParams( target = target )
  dtype = args.dtype
  num_measure_trials = args.num_measure_trials
  timeout = args.timeout
  log_file = args.log_file

  einsum_str = "EFq,FG,WXYZ,Yjk,ZclA,ABo,dgBC,CDp,abcd,UXa,MN,MOS,PRf,efgh,IJs,RJL,ijm,QiK,OQW,kln,hDE,Vbe,STUV,NPT,GHr,HI->KLmnopqrs"
  func = einsum_tree
  sizes = (56, 7, 48, 4, 11, 20, 5, 6, 25, 79, 3, 3, 13, 9, 5, 27, 17, 10, 46, 7, 15, 25, 19, 6, 68, 22, 26, 24, 22, 7, 9, 68, 8, 6, 17, 5, 7, 11, 9, 9, 9, 9, 9, 9, 9)

  tvm_helper.run_all( einsum_str,
                      func,
                      sizes,
                      dtype,
                      hardware_params,
                      target,
                      num_measure_trials,
                      timeout,
                      log_file )
