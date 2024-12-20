import tvm
import tvm.auto_scheduler
import tvm_helper
import argparse
##
# Note: This script was automatically generated by tvm_from_tree.py.
#
# input_string: [[[[[81,87,24,25]->[24,87,81,25]],[[63,87,11]->[11,63,87]]->[11,24,63,81,25]],[[[[[70,75,81],[[77,65,53,70]->[65,77,53,70]]->[65,77,53,75,81]],[[[61,75,22,23]->[22,23,61,75]],[[53,83,61]->[83,53,61]]->[22,23,83,53,75]]->[65,22,23,77,83,81]],[[[[[[66,83,20,21]->[20,21,66,83]],[[5,6,20,21]->[6,5,20,21]]->[6,5,66,83]],[[[54,60,66],[[[[82,76,52,54]->[52,76,82,54]],[[59,76,77]->[77,59,76]]->[77,52,59,82,54]],[[[85,57,82]->[57,85,82]],[[[64,78,85,59]->[64,78,59,85]],[[[79,78,86]->[86,79,78]],[[80,79],[[80,58,64]->[58,64,80]]->[58,64,79]]->[86,58,64,78]]->[86,58,59,85]]->[86,57,58,59,82]]->[86,77,52,57,58,54]]->[86,77,52,57,58,60,66]],[[[[[[71,60,18,19]->[18,19,71,60]],[3,4,18,19]->[3,4,71,60]],[[45,4,5]->[5,45,4]]->[5,3,45,71,60]],[52,84,71]->[52,5,3,45,84,60]],[[[[[[74,84,16,17]->[74,16,17,84]],[1,2,16,17]->[1,74,2,84]],[[[[[[[[72,56,74]->[56,72,74]],[[67,57,62,72]->[57,67,62,72]]->[57,67,56,62,74]],[[[62,69,73]->[69,73,62]],[[73,56,14,15]->[14,15,56,73]]->[14,15,69,56,62]]->[57,14,15,67,69,74]],[[[68,69,12,13]->[12,13,68,69]],[[[55,48,68]->[48,55,68]],[[58,55,67]->[58,67,55]]->[58,48,67,68]]->[58,12,13,48,67,69]]->[57,58,14,15,12,13,48,74]],[[[34,48,49]->[34,49,48]],[[49,50,12,13]->[50,12,13,49]]->[34,50,12,13,48]]->[57,58,34,14,15,50,74]],[[[42,50,51]->[42,51,50]],[[51,0,14,15]->[0,14,15,51]]->[0,14,42,15,50]]->[57,58,0,34,42,74]],[[[[35,36,42,43]->[36,43,35,42]],[[43,0,1]->[0,1,43]]->[36,0,1,35,42]],[28,34,35]->[28,36,0,1,34,42]]->[57,58,28,36,1,74]]->[57,58,28,36,2,84]],[[44,2,3]->[3,44,2]]->[57,3,58,28,36,44,84]],[[[[37,38,44,45]->[45,38,37,44]],[[33,38,39]->[39,33,38]]->[39,45,33,37,44]],[[[32,36,37]->[36,32,37]],[[[29,30,32,33]->[29,30,33,32]],[[[[27,30,31]->[31,27,30]],[26,27]->[31,26,30]],[[26,28,29]->[28,29,26]]->[31,28,29,30]]->[31,28,33,32]]->[31,28,36,33,37]]->[31,39,28,45,36,44]]->[31,39,57,3,58,45,84]]->[31,39,52,57,5,58,60]]->[86,31,39,77,5,66]]->[86,6,31,39,77,83]],[[[[39,40,46,47]->[47,46,40,39]],[[46,6,7]->[7,6,46]]->[7,6,47,40,39]],[[31,40,41]->[41,31,40]]->[41,7,6,47,31,39]]->[41,86,7,47,77,83]],[[[47,8,9]->[9,8,47]],[[7,8,22,23]->[22,23,7,8]]->[9,22,23,7,47]]->[9,41,22,23,86,77,83]]->[9,41,65,86,81]],[[86,65,63]->[63,65,86]]->[9,41,63,81]]->[11,24,9,41,25]],[[9,10,24,25]->[10,24,9,25]]->[11,10,41,25]],[[41,10,11]->[11,10,41]]->[11,25]
##
@tvm.auto_scheduler.register_workload
def einsum_tree( dim_0, dim_1, dim_2, dim_3, dim_4, dim_5, dim_6, dim_7, dim_8, dim_9, dim_10, dim_11, dim_12, dim_13, dim_14, dim_15, dim_16, dim_17, dim_18, dim_19, dim_20, dim_21, dim_22, dim_23, dim_24, dim_25, dim_26, dim_27, dim_28, dim_29, dim_30, dim_31, dim_32, dim_33, dim_34, dim_35, dim_36, dim_37, dim_38, dim_39, dim_40, dim_41, dim_42, dim_43, dim_44, dim_45, dim_46, dim_47, dim_48, dim_49, dim_50, dim_51, dim_52, dim_53, dim_54, dim_55, dim_56, dim_57, dim_58, dim_59, dim_60, dim_61, dim_62, dim_63, dim_64, dim_65, dim_66, dim_67, dim_68, dim_69, dim_70, dim_71, dim_72, dim_73, dim_74, dim_75, dim_76, dim_77, dim_78, dim_79, dim_80, dim_81, dim_82, dim_83, dim_84, dim_85, dim_86, dim_87, dtype):
  tensor_47_8_9 = tvm.te.placeholder((dim_47, dim_8, dim_9), name='tensor_47_8_9', dtype=dtype)
  tensor_7_8_22_23 = tvm.te.placeholder((dim_7, dim_8, dim_22, dim_23), name='tensor_7_8_22_23', dtype=dtype)
  tensor_39_40_46_47 = tvm.te.placeholder((dim_39, dim_40, dim_46, dim_47), name='tensor_39_40_46_47', dtype=dtype)
  tensor_46_6_7 = tvm.te.placeholder((dim_46, dim_6, dim_7), name='tensor_46_6_7', dtype=dtype)
  tensor_27_30_31 = tvm.te.placeholder((dim_27, dim_30, dim_31), name='tensor_27_30_31', dtype=dtype)
  tensor_26_27 = tvm.te.placeholder((dim_26, dim_27), name='tensor_26_27', dtype=dtype)
  tensor_37_38_44_45 = tvm.te.placeholder((dim_37, dim_38, dim_44, dim_45), name='tensor_37_38_44_45', dtype=dtype)
  tensor_33_38_39 = tvm.te.placeholder((dim_33, dim_38, dim_39), name='tensor_33_38_39', dtype=dtype)
  tensor_35_36_42_43 = tvm.te.placeholder((dim_35, dim_36, dim_42, dim_43), name='tensor_35_36_42_43', dtype=dtype)
  tensor_43_0_1 = tvm.te.placeholder((dim_43, dim_0, dim_1), name='tensor_43_0_1', dtype=dtype)
  tensor_42_50_51 = tvm.te.placeholder((dim_42, dim_50, dim_51), name='tensor_42_50_51', dtype=dtype)
  tensor_51_0_14_15 = tvm.te.placeholder((dim_51, dim_0, dim_14, dim_15), name='tensor_51_0_14_15', dtype=dtype)
  tensor_34_48_49 = tvm.te.placeholder((dim_34, dim_48, dim_49), name='tensor_34_48_49', dtype=dtype)
  tensor_49_50_12_13 = tvm.te.placeholder((dim_49, dim_50, dim_12, dim_13), name='tensor_49_50_12_13', dtype=dtype)
  tensor_55_48_68 = tvm.te.placeholder((dim_55, dim_48, dim_68), name='tensor_55_48_68', dtype=dtype)
  tensor_58_55_67 = tvm.te.placeholder((dim_58, dim_55, dim_67), name='tensor_58_55_67', dtype=dtype)
  tensor_62_69_73 = tvm.te.placeholder((dim_62, dim_69, dim_73), name='tensor_62_69_73', dtype=dtype)
  tensor_73_56_14_15 = tvm.te.placeholder((dim_73, dim_56, dim_14, dim_15), name='tensor_73_56_14_15', dtype=dtype)
  tensor_72_56_74 = tvm.te.placeholder((dim_72, dim_56, dim_74), name='tensor_72_56_74', dtype=dtype)
  tensor_67_57_62_72 = tvm.te.placeholder((dim_67, dim_57, dim_62, dim_72), name='tensor_67_57_62_72', dtype=dtype)
  tensor_74_84_16_17 = tvm.te.placeholder((dim_74, dim_84, dim_16, dim_17), name='tensor_74_84_16_17', dtype=dtype)
  tensor_1_2_16_17 = tvm.te.placeholder((dim_1, dim_2, dim_16, dim_17), name='tensor_1_2_16_17', dtype=dtype)
  tensor_71_60_18_19 = tvm.te.placeholder((dim_71, dim_60, dim_18, dim_19), name='tensor_71_60_18_19', dtype=dtype)
  tensor_3_4_18_19 = tvm.te.placeholder((dim_3, dim_4, dim_18, dim_19), name='tensor_3_4_18_19', dtype=dtype)
  tensor_80_79 = tvm.te.placeholder((dim_80, dim_79), name='tensor_80_79', dtype=dtype)
  tensor_80_58_64 = tvm.te.placeholder((dim_80, dim_58, dim_64), name='tensor_80_58_64', dtype=dtype)
  tensor_82_76_52_54 = tvm.te.placeholder((dim_82, dim_76, dim_52, dim_54), name='tensor_82_76_52_54', dtype=dtype)
  tensor_59_76_77 = tvm.te.placeholder((dim_59, dim_76, dim_77), name='tensor_59_76_77', dtype=dtype)
  tensor_66_83_20_21 = tvm.te.placeholder((dim_66, dim_83, dim_20, dim_21), name='tensor_66_83_20_21', dtype=dtype)
  tensor_5_6_20_21 = tvm.te.placeholder((dim_5, dim_6, dim_20, dim_21), name='tensor_5_6_20_21', dtype=dtype)
  tensor_61_75_22_23 = tvm.te.placeholder((dim_61, dim_75, dim_22, dim_23), name='tensor_61_75_22_23', dtype=dtype)
  tensor_53_83_61 = tvm.te.placeholder((dim_53, dim_83, dim_61), name='tensor_53_83_61', dtype=dtype)
  tensor_70_75_81 = tvm.te.placeholder((dim_70, dim_75, dim_81), name='tensor_70_75_81', dtype=dtype)
  tensor_77_65_53_70 = tvm.te.placeholder((dim_77, dim_65, dim_53, dim_70), name='tensor_77_65_53_70', dtype=dtype)
  tensor_81_87_24_25 = tvm.te.placeholder((dim_81, dim_87, dim_24, dim_25), name='tensor_81_87_24_25', dtype=dtype)
  tensor_63_87_11 = tvm.te.placeholder((dim_63, dim_87, dim_11), name='tensor_63_87_11', dtype=dtype)
  tensor_41_10_11 = tvm.te.placeholder((dim_41, dim_10, dim_11), name='tensor_41_10_11', dtype=dtype)
  tensor_9_10_24_25 = tvm.te.placeholder((dim_9, dim_10, dim_24, dim_25), name='tensor_9_10_24_25', dtype=dtype)
  tensor_86_65_63 = tvm.te.placeholder((dim_86, dim_65, dim_63), name='tensor_86_65_63', dtype=dtype)
  tensor_31_40_41 = tvm.te.placeholder((dim_31, dim_40, dim_41), name='tensor_31_40_41', dtype=dtype)
  tensor_26_28_29 = tvm.te.placeholder((dim_26, dim_28, dim_29), name='tensor_26_28_29', dtype=dtype)
  tensor_29_30_32_33 = tvm.te.placeholder((dim_29, dim_30, dim_32, dim_33), name='tensor_29_30_32_33', dtype=dtype)
  tensor_32_36_37 = tvm.te.placeholder((dim_32, dim_36, dim_37), name='tensor_32_36_37', dtype=dtype)
  tensor_44_2_3 = tvm.te.placeholder((dim_44, dim_2, dim_3), name='tensor_44_2_3', dtype=dtype)
  tensor_28_34_35 = tvm.te.placeholder((dim_28, dim_34, dim_35), name='tensor_28_34_35', dtype=dtype)
  tensor_68_69_12_13 = tvm.te.placeholder((dim_68, dim_69, dim_12, dim_13), name='tensor_68_69_12_13', dtype=dtype)
  tensor_52_84_71 = tvm.te.placeholder((dim_52, dim_84, dim_71), name='tensor_52_84_71', dtype=dtype)
  tensor_45_4_5 = tvm.te.placeholder((dim_45, dim_4, dim_5), name='tensor_45_4_5', dtype=dtype)
  tensor_79_78_86 = tvm.te.placeholder((dim_79, dim_78, dim_86), name='tensor_79_78_86', dtype=dtype)
  tensor_64_78_85_59 = tvm.te.placeholder((dim_64, dim_78, dim_85, dim_59), name='tensor_64_78_85_59', dtype=dtype)
  tensor_85_57_82 = tvm.te.placeholder((dim_85, dim_57, dim_82), name='tensor_85_57_82', dtype=dtype)
  tensor_54_60_66 = tvm.te.placeholder((dim_54, dim_60, dim_66), name='tensor_54_60_66', dtype=dtype)

  tmp_45 = tvm.te.reduce_axis((0, dim_45), name='tmp_45')
  tmp_82 = tvm.te.reduce_axis((0, dim_82), name='tmp_82')
  tmp_40 = tvm.te.reduce_axis((0, dim_40), name='tmp_40')
  tmp_19 = tvm.te.reduce_axis((0, dim_19), name='tmp_19')
  tmp_36 = tvm.te.reduce_axis((0, dim_36), name='tmp_36')
  tmp_73 = tvm.te.reduce_axis((0, dim_73), name='tmp_73')
  tmp_30 = tvm.te.reduce_axis((0, dim_30), name='tmp_30')
  tmp_26 = tvm.te.reduce_axis((0, dim_26), name='tmp_26')
  tmp_72 = tvm.te.reduce_axis((0, dim_72), name='tmp_72')
  tmp_54 = tvm.te.reduce_axis((0, dim_54), name='tmp_54')
  tmp_76 = tvm.te.reduce_axis((0, dim_76), name='tmp_76')
  tmp_57 = tvm.te.reduce_axis((0, dim_57), name='tmp_57')
  tmp_17 = tvm.te.reduce_axis((0, dim_17), name='tmp_17')
  tmp_13 = tvm.te.reduce_axis((0, dim_13), name='tmp_13')
  tmp_79 = tvm.te.reduce_axis((0, dim_79), name='tmp_79')
  tmp_68 = tvm.te.reduce_axis((0, dim_68), name='tmp_68')
  tmp_61 = tvm.te.reduce_axis((0, dim_61), name='tmp_61')
  tmp_33 = tvm.te.reduce_axis((0, dim_33), name='tmp_33')
  tmp_46 = tvm.te.reduce_axis((0, dim_46), name='tmp_46')
  tmp_80 = tvm.te.reduce_axis((0, dim_80), name='tmp_80')
  tmp_28 = tvm.te.reduce_axis((0, dim_28), name='tmp_28')
  tmp_6 = tvm.te.reduce_axis((0, dim_6), name='tmp_6')
  tmp_23 = tvm.te.reduce_axis((0, dim_23), name='tmp_23')
  tmp_1 = tvm.te.reduce_axis((0, dim_1), name='tmp_1')
  tmp_18 = tvm.te.reduce_axis((0, dim_18), name='tmp_18')
  tmp_63 = tvm.te.reduce_axis((0, dim_63), name='tmp_63')
  tmp_0 = tvm.te.reduce_axis((0, dim_0), name='tmp_0')
  tmp_75 = tvm.te.reduce_axis((0, dim_75), name='tmp_75')
  tmp_67 = tvm.te.reduce_axis((0, dim_67), name='tmp_67')
  tmp_49 = tvm.te.reduce_axis((0, dim_49), name='tmp_49')
  tmp_34 = tvm.te.reduce_axis((0, dim_34), name='tmp_34')
  tmp_64 = tvm.te.reduce_axis((0, dim_64), name='tmp_64')
  tmp_20 = tvm.te.reduce_axis((0, dim_20), name='tmp_20')
  tmp_47 = tvm.te.reduce_axis((0, dim_47), name='tmp_47')
  tmp_65 = tvm.te.reduce_axis((0, dim_65), name='tmp_65')
  tmp_4 = tvm.te.reduce_axis((0, dim_4), name='tmp_4')
  tmp_29 = tvm.te.reduce_axis((0, dim_29), name='tmp_29')
  tmp_5 = tvm.te.reduce_axis((0, dim_5), name='tmp_5')
  tmp_21 = tvm.te.reduce_axis((0, dim_21), name='tmp_21')
  tmp_3 = tvm.te.reduce_axis((0, dim_3), name='tmp_3')
  tmp_22 = tvm.te.reduce_axis((0, dim_22), name='tmp_22')
  tmp_48 = tvm.te.reduce_axis((0, dim_48), name='tmp_48')
  tmp_81 = tvm.te.reduce_axis((0, dim_81), name='tmp_81')
  tmp_71 = tvm.te.reduce_axis((0, dim_71), name='tmp_71')
  tmp_8 = tvm.te.reduce_axis((0, dim_8), name='tmp_8')
  tmp_10 = tvm.te.reduce_axis((0, dim_10), name='tmp_10')
  tmp_53 = tvm.te.reduce_axis((0, dim_53), name='tmp_53')
  tmp_74 = tvm.te.reduce_axis((0, dim_74), name='tmp_74')
  tmp_58 = tvm.te.reduce_axis((0, dim_58), name='tmp_58')
  tmp_15 = tvm.te.reduce_axis((0, dim_15), name='tmp_15')
  tmp_38 = tvm.te.reduce_axis((0, dim_38), name='tmp_38')
  tmp_41 = tvm.te.reduce_axis((0, dim_41), name='tmp_41')
  tmp_78 = tvm.te.reduce_axis((0, dim_78), name='tmp_78')
  tmp_31 = tvm.te.reduce_axis((0, dim_31), name='tmp_31')
  tmp_9 = tvm.te.reduce_axis((0, dim_9), name='tmp_9')
  tmp_39 = tvm.te.reduce_axis((0, dim_39), name='tmp_39')
  tmp_37 = tvm.te.reduce_axis((0, dim_37), name='tmp_37')
  tmp_83 = tvm.te.reduce_axis((0, dim_83), name='tmp_83')
  tmp_14 = tvm.te.reduce_axis((0, dim_14), name='tmp_14')
  tmp_84 = tvm.te.reduce_axis((0, dim_84), name='tmp_84')
  tmp_70 = tvm.te.reduce_axis((0, dim_70), name='tmp_70')
  tmp_32 = tvm.te.reduce_axis((0, dim_32), name='tmp_32')
  tmp_16 = tvm.te.reduce_axis((0, dim_16), name='tmp_16')
  tmp_35 = tvm.te.reduce_axis((0, dim_35), name='tmp_35')
  tmp_51 = tvm.te.reduce_axis((0, dim_51), name='tmp_51')
  tmp_44 = tvm.te.reduce_axis((0, dim_44), name='tmp_44')
  tmp_69 = tvm.te.reduce_axis((0, dim_69), name='tmp_69')
  tmp_50 = tvm.te.reduce_axis((0, dim_50), name='tmp_50')
  tmp_62 = tvm.te.reduce_axis((0, dim_62), name='tmp_62')
  tmp_52 = tvm.te.reduce_axis((0, dim_52), name='tmp_52')
  tmp_27 = tvm.te.reduce_axis((0, dim_27), name='tmp_27')
  tmp_86 = tvm.te.reduce_axis((0, dim_86), name='tmp_86')
  tmp_12 = tvm.te.reduce_axis((0, dim_12), name='tmp_12')
  tmp_2 = tvm.te.reduce_axis((0, dim_2), name='tmp_2')
  tmp_60 = tvm.te.reduce_axis((0, dim_60), name='tmp_60')
  tmp_7 = tvm.te.reduce_axis((0, dim_7), name='tmp_7')
  tmp_43 = tvm.te.reduce_axis((0, dim_43), name='tmp_43')
  tmp_59 = tvm.te.reduce_axis((0, dim_59), name='tmp_59')
  tmp_56 = tvm.te.reduce_axis((0, dim_56), name='tmp_56')
  tmp_42 = tvm.te.reduce_axis((0, dim_42), name='tmp_42')
  tmp_85 = tvm.te.reduce_axis((0, dim_85), name='tmp_85')
  tmp_55 = tvm.te.reduce_axis((0, dim_55), name='tmp_55')
  tmp_77 = tvm.te.reduce_axis((0, dim_77), name='tmp_77')
  tmp_24 = tvm.te.reduce_axis((0, dim_24), name='tmp_24')
  tmp_87 = tvm.te.reduce_axis((0, dim_87), name='tmp_87')
  tmp_66 = tvm.te.reduce_axis((0, dim_66), name='tmp_66')

  tensor_9_22_23_7_47 = tvm.te.compute( (dim_9, dim_22, dim_23, dim_7, dim_47), lambda tmp_9, tmp_22, tmp_23, tmp_7, tmp_47: tvm.te.sum( tensor_47_8_9[ tmp_47, tmp_8, tmp_9 ] * tensor_7_8_22_23[ tmp_7, tmp_8, tmp_22, tmp_23 ] , axis=[ tmp_8 ]), name='tensor_9_22_23_7_47' )
  tensor_7_6_47_40_39 = tvm.te.compute( (dim_7, dim_6, dim_47, dim_40, dim_39), lambda tmp_7, tmp_6, tmp_47, tmp_40, tmp_39: tvm.te.sum( tensor_39_40_46_47[ tmp_39, tmp_40, tmp_46, tmp_47 ] * tensor_46_6_7[ tmp_46, tmp_6, tmp_7 ] , axis=[ tmp_46 ]), name='tensor_7_6_47_40_39' )
  tensor_41_7_6_47_31_39 = tvm.te.compute( (dim_41, dim_7, dim_6, dim_47, dim_31, dim_39), lambda tmp_41, tmp_7, tmp_6, tmp_47, tmp_31, tmp_39: tvm.te.sum( tensor_7_6_47_40_39[ tmp_7, tmp_6, tmp_47, tmp_40, tmp_39 ] * tensor_31_40_41[ tmp_31, tmp_40, tmp_41 ] , axis=[ tmp_40 ]), name='tensor_41_7_6_47_31_39' )
  tensor_31_26_30 = tvm.te.compute( (dim_31, dim_26, dim_30), lambda tmp_31, tmp_26, tmp_30: tvm.te.sum( tensor_27_30_31[ tmp_27, tmp_30, tmp_31 ] * tensor_26_27[ tmp_26, tmp_27 ] , axis=[ tmp_27 ]), name='tensor_31_26_30' )
  tensor_31_28_29_30 = tvm.te.compute( (dim_31, dim_28, dim_29, dim_30), lambda tmp_31, tmp_28, tmp_29, tmp_30: tvm.te.sum( tensor_31_26_30[ tmp_31, tmp_26, tmp_30 ] * tensor_26_28_29[ tmp_26, tmp_28, tmp_29 ] , axis=[ tmp_26 ]), name='tensor_31_28_29_30' )
  tensor_31_28_33_32 = tvm.te.compute( (dim_31, dim_28, dim_33, dim_32), lambda tmp_31, tmp_28, tmp_33, tmp_32: tvm.te.sum( tensor_29_30_32_33[ tmp_29, tmp_30, tmp_32, tmp_33 ] * tensor_31_28_29_30[ tmp_31, tmp_28, tmp_29, tmp_30 ] , axis=[ tmp_29, tmp_30 ]), name='tensor_31_28_33_32' )
  tensor_31_28_36_33_37 = tvm.te.compute( (dim_31, dim_28, dim_36, dim_33, dim_37), lambda tmp_31, tmp_28, tmp_36, tmp_33, tmp_37: tvm.te.sum( tensor_32_36_37[ tmp_32, tmp_36, tmp_37 ] * tensor_31_28_33_32[ tmp_31, tmp_28, tmp_33, tmp_32 ] , axis=[ tmp_32 ]), name='tensor_31_28_36_33_37' )
  tensor_39_45_33_37_44 = tvm.te.compute( (dim_39, dim_45, dim_33, dim_37, dim_44), lambda tmp_39, tmp_45, tmp_33, tmp_37, tmp_44: tvm.te.sum( tensor_37_38_44_45[ tmp_37, tmp_38, tmp_44, tmp_45 ] * tensor_33_38_39[ tmp_33, tmp_38, tmp_39 ] , axis=[ tmp_38 ]), name='tensor_39_45_33_37_44' )
  tensor_31_39_28_45_36_44 = tvm.te.compute( (dim_31, dim_39, dim_28, dim_45, dim_36, dim_44), lambda tmp_31, tmp_39, tmp_28, tmp_45, tmp_36, tmp_44: tvm.te.sum( tensor_39_45_33_37_44[ tmp_39, tmp_45, tmp_33, tmp_37, tmp_44 ] * tensor_31_28_36_33_37[ tmp_31, tmp_28, tmp_36, tmp_33, tmp_37 ] , axis=[ tmp_33, tmp_37 ]), name='tensor_31_39_28_45_36_44' )
  tensor_36_0_1_35_42 = tvm.te.compute( (dim_36, dim_0, dim_1, dim_35, dim_42), lambda tmp_36, tmp_0, tmp_1, tmp_35, tmp_42: tvm.te.sum( tensor_35_36_42_43[ tmp_35, tmp_36, tmp_42, tmp_43 ] * tensor_43_0_1[ tmp_43, tmp_0, tmp_1 ] , axis=[ tmp_43 ]), name='tensor_36_0_1_35_42' )
  tensor_28_36_0_1_34_42 = tvm.te.compute( (dim_28, dim_36, dim_0, dim_1, dim_34, dim_42), lambda tmp_28, tmp_36, tmp_0, tmp_1, tmp_34, tmp_42: tvm.te.sum( tensor_36_0_1_35_42[ tmp_36, tmp_0, tmp_1, tmp_35, tmp_42 ] * tensor_28_34_35[ tmp_28, tmp_34, tmp_35 ] , axis=[ tmp_35 ]), name='tensor_28_36_0_1_34_42' )
  tensor_0_14_42_15_50 = tvm.te.compute( (dim_0, dim_14, dim_42, dim_15, dim_50), lambda tmp_0, tmp_14, tmp_42, tmp_15, tmp_50: tvm.te.sum( tensor_42_50_51[ tmp_42, tmp_50, tmp_51 ] * tensor_51_0_14_15[ tmp_51, tmp_0, tmp_14, tmp_15 ] , axis=[ tmp_51 ]), name='tensor_0_14_42_15_50' )
  tensor_34_50_12_13_48 = tvm.te.compute( (dim_34, dim_50, dim_12, dim_13, dim_48), lambda tmp_34, tmp_50, tmp_12, tmp_13, tmp_48: tvm.te.sum( tensor_34_48_49[ tmp_34, tmp_48, tmp_49 ] * tensor_49_50_12_13[ tmp_49, tmp_50, tmp_12, tmp_13 ] , axis=[ tmp_49 ]), name='tensor_34_50_12_13_48' )
  tensor_58_48_67_68 = tvm.te.compute( (dim_58, dim_48, dim_67, dim_68), lambda tmp_58, tmp_48, tmp_67, tmp_68: tvm.te.sum( tensor_55_48_68[ tmp_55, tmp_48, tmp_68 ] * tensor_58_55_67[ tmp_58, tmp_55, tmp_67 ] , axis=[ tmp_55 ]), name='tensor_58_48_67_68' )
  tensor_58_12_13_48_67_69 = tvm.te.compute( (dim_58, dim_12, dim_13, dim_48, dim_67, dim_69), lambda tmp_58, tmp_12, tmp_13, tmp_48, tmp_67, tmp_69: tvm.te.sum( tensor_68_69_12_13[ tmp_68, tmp_69, tmp_12, tmp_13 ] * tensor_58_48_67_68[ tmp_58, tmp_48, tmp_67, tmp_68 ] , axis=[ tmp_68 ]), name='tensor_58_12_13_48_67_69' )
  tensor_14_15_69_56_62 = tvm.te.compute( (dim_14, dim_15, dim_69, dim_56, dim_62), lambda tmp_14, tmp_15, tmp_69, tmp_56, tmp_62: tvm.te.sum( tensor_62_69_73[ tmp_62, tmp_69, tmp_73 ] * tensor_73_56_14_15[ tmp_73, tmp_56, tmp_14, tmp_15 ] , axis=[ tmp_73 ]), name='tensor_14_15_69_56_62' )
  tensor_57_67_56_62_74 = tvm.te.compute( (dim_57, dim_67, dim_56, dim_62, dim_74), lambda tmp_57, tmp_67, tmp_56, tmp_62, tmp_74: tvm.te.sum( tensor_72_56_74[ tmp_72, tmp_56, tmp_74 ] * tensor_67_57_62_72[ tmp_67, tmp_57, tmp_62, tmp_72 ] , axis=[ tmp_72 ]), name='tensor_57_67_56_62_74' )
  tensor_57_14_15_67_69_74 = tvm.te.compute( (dim_57, dim_14, dim_15, dim_67, dim_69, dim_74), lambda tmp_57, tmp_14, tmp_15, tmp_67, tmp_69, tmp_74: tvm.te.sum( tensor_57_67_56_62_74[ tmp_57, tmp_67, tmp_56, tmp_62, tmp_74 ] * tensor_14_15_69_56_62[ tmp_14, tmp_15, tmp_69, tmp_56, tmp_62 ] , axis=[ tmp_62, tmp_56 ]), name='tensor_57_14_15_67_69_74' )
  tensor_57_58_14_15_12_13_48_74 = tvm.te.compute( (dim_57, dim_58, dim_14, dim_15, dim_12, dim_13, dim_48, dim_74), lambda tmp_57, tmp_58, tmp_14, tmp_15, tmp_12, tmp_13, tmp_48, tmp_74: tvm.te.sum( tensor_57_14_15_67_69_74[ tmp_57, tmp_14, tmp_15, tmp_67, tmp_69, tmp_74 ] * tensor_58_12_13_48_67_69[ tmp_58, tmp_12, tmp_13, tmp_48, tmp_67, tmp_69 ] , axis=[ tmp_67, tmp_69 ]), name='tensor_57_58_14_15_12_13_48_74' )
  tensor_57_58_34_14_15_50_74 = tvm.te.compute( (dim_57, dim_58, dim_34, dim_14, dim_15, dim_50, dim_74), lambda tmp_57, tmp_58, tmp_34, tmp_14, tmp_15, tmp_50, tmp_74: tvm.te.sum( tensor_57_58_14_15_12_13_48_74[ tmp_57, tmp_58, tmp_14, tmp_15, tmp_12, tmp_13, tmp_48, tmp_74 ] * tensor_34_50_12_13_48[ tmp_34, tmp_50, tmp_12, tmp_13, tmp_48 ] , axis=[ tmp_48, tmp_13, tmp_12 ]), name='tensor_57_58_34_14_15_50_74' )
  tensor_57_58_0_34_42_74 = tvm.te.compute( (dim_57, dim_58, dim_0, dim_34, dim_42, dim_74), lambda tmp_57, tmp_58, tmp_0, tmp_34, tmp_42, tmp_74: tvm.te.sum( tensor_57_58_34_14_15_50_74[ tmp_57, tmp_58, tmp_34, tmp_14, tmp_15, tmp_50, tmp_74 ] * tensor_0_14_42_15_50[ tmp_0, tmp_14, tmp_42, tmp_15, tmp_50 ] , axis=[ tmp_15, tmp_14, tmp_50 ]), name='tensor_57_58_0_34_42_74' )
  tensor_57_58_28_36_1_74 = tvm.te.compute( (dim_57, dim_58, dim_28, dim_36, dim_1, dim_74), lambda tmp_57, tmp_58, tmp_28, tmp_36, tmp_1, tmp_74: tvm.te.sum( tensor_57_58_0_34_42_74[ tmp_57, tmp_58, tmp_0, tmp_34, tmp_42, tmp_74 ] * tensor_28_36_0_1_34_42[ tmp_28, tmp_36, tmp_0, tmp_1, tmp_34, tmp_42 ] , axis=[ tmp_0, tmp_34, tmp_42 ]), name='tensor_57_58_28_36_1_74' )
  tensor_1_74_2_84 = tvm.te.compute( (dim_1, dim_74, dim_2, dim_84), lambda tmp_1, tmp_74, tmp_2, tmp_84: tvm.te.sum( tensor_74_84_16_17[ tmp_74, tmp_84, tmp_16, tmp_17 ] * tensor_1_2_16_17[ tmp_1, tmp_2, tmp_16, tmp_17 ] , axis=[ tmp_17, tmp_16 ]), name='tensor_1_74_2_84' )
  tensor_57_58_28_36_2_84 = tvm.te.compute( (dim_57, dim_58, dim_28, dim_36, dim_2, dim_84), lambda tmp_57, tmp_58, tmp_28, tmp_36, tmp_2, tmp_84: tvm.te.sum( tensor_1_74_2_84[ tmp_1, tmp_74, tmp_2, tmp_84 ] * tensor_57_58_28_36_1_74[ tmp_57, tmp_58, tmp_28, tmp_36, tmp_1, tmp_74 ] , axis=[ tmp_74, tmp_1 ]), name='tensor_57_58_28_36_2_84' )
  tensor_57_3_58_28_36_44_84 = tvm.te.compute( (dim_57, dim_3, dim_58, dim_28, dim_36, dim_44, dim_84), lambda tmp_57, tmp_3, tmp_58, tmp_28, tmp_36, tmp_44, tmp_84: tvm.te.sum( tensor_57_58_28_36_2_84[ tmp_57, tmp_58, tmp_28, tmp_36, tmp_2, tmp_84 ] * tensor_44_2_3[ tmp_44, tmp_2, tmp_3 ] , axis=[ tmp_2 ]), name='tensor_57_3_58_28_36_44_84' )
  tensor_31_39_57_3_58_45_84 = tvm.te.compute( (dim_31, dim_39, dim_57, dim_3, dim_58, dim_45, dim_84), lambda tmp_31, tmp_39, tmp_57, tmp_3, tmp_58, tmp_45, tmp_84: tvm.te.sum( tensor_57_3_58_28_36_44_84[ tmp_57, tmp_3, tmp_58, tmp_28, tmp_36, tmp_44, tmp_84 ] * tensor_31_39_28_45_36_44[ tmp_31, tmp_39, tmp_28, tmp_45, tmp_36, tmp_44 ] , axis=[ tmp_44, tmp_28, tmp_36 ]), name='tensor_31_39_57_3_58_45_84' )
  tensor_3_4_71_60 = tvm.te.compute( (dim_3, dim_4, dim_71, dim_60), lambda tmp_3, tmp_4, tmp_71, tmp_60: tvm.te.sum( tensor_71_60_18_19[ tmp_71, tmp_60, tmp_18, tmp_19 ] * tensor_3_4_18_19[ tmp_3, tmp_4, tmp_18, tmp_19 ] , axis=[ tmp_19, tmp_18 ]), name='tensor_3_4_71_60' )
  tensor_5_3_45_71_60 = tvm.te.compute( (dim_5, dim_3, dim_45, dim_71, dim_60), lambda tmp_5, tmp_3, tmp_45, tmp_71, tmp_60: tvm.te.sum( tensor_3_4_71_60[ tmp_3, tmp_4, tmp_71, tmp_60 ] * tensor_45_4_5[ tmp_45, tmp_4, tmp_5 ] , axis=[ tmp_4 ]), name='tensor_5_3_45_71_60' )
  tensor_52_5_3_45_84_60 = tvm.te.compute( (dim_52, dim_5, dim_3, dim_45, dim_84, dim_60), lambda tmp_52, tmp_5, tmp_3, tmp_45, tmp_84, tmp_60: tvm.te.sum( tensor_5_3_45_71_60[ tmp_5, tmp_3, tmp_45, tmp_71, tmp_60 ] * tensor_52_84_71[ tmp_52, tmp_84, tmp_71 ] , axis=[ tmp_71 ]), name='tensor_52_5_3_45_84_60' )
  tensor_31_39_52_57_5_58_60 = tvm.te.compute( (dim_31, dim_39, dim_52, dim_57, dim_5, dim_58, dim_60), lambda tmp_31, tmp_39, tmp_52, tmp_57, tmp_5, tmp_58, tmp_60: tvm.te.sum( tensor_52_5_3_45_84_60[ tmp_52, tmp_5, tmp_3, tmp_45, tmp_84, tmp_60 ] * tensor_31_39_57_3_58_45_84[ tmp_31, tmp_39, tmp_57, tmp_3, tmp_58, tmp_45, tmp_84 ] , axis=[ tmp_45, tmp_3, tmp_84 ]), name='tensor_31_39_52_57_5_58_60' )
  tensor_58_64_79 = tvm.te.compute( (dim_58, dim_64, dim_79), lambda tmp_58, tmp_64, tmp_79: tvm.te.sum( tensor_80_79[ tmp_80, tmp_79 ] * tensor_80_58_64[ tmp_80, tmp_58, tmp_64 ] , axis=[ tmp_80 ]), name='tensor_58_64_79' )
  tensor_86_58_64_78 = tvm.te.compute( (dim_86, dim_58, dim_64, dim_78), lambda tmp_86, tmp_58, tmp_64, tmp_78: tvm.te.sum( tensor_79_78_86[ tmp_79, tmp_78, tmp_86 ] * tensor_58_64_79[ tmp_58, tmp_64, tmp_79 ] , axis=[ tmp_79 ]), name='tensor_86_58_64_78' )
  tensor_86_58_59_85 = tvm.te.compute( (dim_86, dim_58, dim_59, dim_85), lambda tmp_86, tmp_58, tmp_59, tmp_85: tvm.te.sum( tensor_64_78_85_59[ tmp_64, tmp_78, tmp_85, tmp_59 ] * tensor_86_58_64_78[ tmp_86, tmp_58, tmp_64, tmp_78 ] , axis=[ tmp_64, tmp_78 ]), name='tensor_86_58_59_85' )
  tensor_86_57_58_59_82 = tvm.te.compute( (dim_86, dim_57, dim_58, dim_59, dim_82), lambda tmp_86, tmp_57, tmp_58, tmp_59, tmp_82: tvm.te.sum( tensor_85_57_82[ tmp_85, tmp_57, tmp_82 ] * tensor_86_58_59_85[ tmp_86, tmp_58, tmp_59, tmp_85 ] , axis=[ tmp_85 ]), name='tensor_86_57_58_59_82' )
  tensor_77_52_59_82_54 = tvm.te.compute( (dim_77, dim_52, dim_59, dim_82, dim_54), lambda tmp_77, tmp_52, tmp_59, tmp_82, tmp_54: tvm.te.sum( tensor_82_76_52_54[ tmp_82, tmp_76, tmp_52, tmp_54 ] * tensor_59_76_77[ tmp_59, tmp_76, tmp_77 ] , axis=[ tmp_76 ]), name='tensor_77_52_59_82_54' )
  tensor_86_77_52_57_58_54 = tvm.te.compute( (dim_86, dim_77, dim_52, dim_57, dim_58, dim_54), lambda tmp_86, tmp_77, tmp_52, tmp_57, tmp_58, tmp_54: tvm.te.sum( tensor_77_52_59_82_54[ tmp_77, tmp_52, tmp_59, tmp_82, tmp_54 ] * tensor_86_57_58_59_82[ tmp_86, tmp_57, tmp_58, tmp_59, tmp_82 ] , axis=[ tmp_59, tmp_82 ]), name='tensor_86_77_52_57_58_54' )
  tensor_86_77_52_57_58_60_66 = tvm.te.compute( (dim_86, dim_77, dim_52, dim_57, dim_58, dim_60, dim_66), lambda tmp_86, tmp_77, tmp_52, tmp_57, tmp_58, tmp_60, tmp_66: tvm.te.sum( tensor_54_60_66[ tmp_54, tmp_60, tmp_66 ] * tensor_86_77_52_57_58_54[ tmp_86, tmp_77, tmp_52, tmp_57, tmp_58, tmp_54 ] , axis=[ tmp_54 ]), name='tensor_86_77_52_57_58_60_66' )
  tensor_86_31_39_77_5_66 = tvm.te.compute( (dim_86, dim_31, dim_39, dim_77, dim_5, dim_66), lambda tmp_86, tmp_31, tmp_39, tmp_77, tmp_5, tmp_66: tvm.te.sum( tensor_86_77_52_57_58_60_66[ tmp_86, tmp_77, tmp_52, tmp_57, tmp_58, tmp_60, tmp_66 ] * tensor_31_39_52_57_5_58_60[ tmp_31, tmp_39, tmp_52, tmp_57, tmp_5, tmp_58, tmp_60 ] , axis=[ tmp_57, tmp_60, tmp_58, tmp_52 ]), name='tensor_86_31_39_77_5_66' )
  tensor_6_5_66_83 = tvm.te.compute( (dim_6, dim_5, dim_66, dim_83), lambda tmp_6, tmp_5, tmp_66, tmp_83: tvm.te.sum( tensor_66_83_20_21[ tmp_66, tmp_83, tmp_20, tmp_21 ] * tensor_5_6_20_21[ tmp_5, tmp_6, tmp_20, tmp_21 ] , axis=[ tmp_20, tmp_21 ]), name='tensor_6_5_66_83' )
  tensor_86_6_31_39_77_83 = tvm.te.compute( (dim_86, dim_6, dim_31, dim_39, dim_77, dim_83), lambda tmp_86, tmp_6, tmp_31, tmp_39, tmp_77, tmp_83: tvm.te.sum( tensor_6_5_66_83[ tmp_6, tmp_5, tmp_66, tmp_83 ] * tensor_86_31_39_77_5_66[ tmp_86, tmp_31, tmp_39, tmp_77, tmp_5, tmp_66 ] , axis=[ tmp_66, tmp_5 ]), name='tensor_86_6_31_39_77_83' )
  tensor_41_86_7_47_77_83 = tvm.te.compute( (dim_41, dim_86, dim_7, dim_47, dim_77, dim_83), lambda tmp_41, tmp_86, tmp_7, tmp_47, tmp_77, tmp_83: tvm.te.sum( tensor_86_6_31_39_77_83[ tmp_86, tmp_6, tmp_31, tmp_39, tmp_77, tmp_83 ] * tensor_41_7_6_47_31_39[ tmp_41, tmp_7, tmp_6, tmp_47, tmp_31, tmp_39 ] , axis=[ tmp_31, tmp_39, tmp_6 ]), name='tensor_41_86_7_47_77_83' )
  tensor_9_41_22_23_86_77_83 = tvm.te.compute( (dim_9, dim_41, dim_22, dim_23, dim_86, dim_77, dim_83), lambda tmp_9, tmp_41, tmp_22, tmp_23, tmp_86, tmp_77, tmp_83: tvm.te.sum( tensor_41_86_7_47_77_83[ tmp_41, tmp_86, tmp_7, tmp_47, tmp_77, tmp_83 ] * tensor_9_22_23_7_47[ tmp_9, tmp_22, tmp_23, tmp_7, tmp_47 ] , axis=[ tmp_7, tmp_47 ]), name='tensor_9_41_22_23_86_77_83' )
  tensor_22_23_83_53_75 = tvm.te.compute( (dim_22, dim_23, dim_83, dim_53, dim_75), lambda tmp_22, tmp_23, tmp_83, tmp_53, tmp_75: tvm.te.sum( tensor_61_75_22_23[ tmp_61, tmp_75, tmp_22, tmp_23 ] * tensor_53_83_61[ tmp_53, tmp_83, tmp_61 ] , axis=[ tmp_61 ]), name='tensor_22_23_83_53_75' )
  tensor_65_77_53_75_81 = tvm.te.compute( (dim_65, dim_77, dim_53, dim_75, dim_81), lambda tmp_65, tmp_77, tmp_53, tmp_75, tmp_81: tvm.te.sum( tensor_70_75_81[ tmp_70, tmp_75, tmp_81 ] * tensor_77_65_53_70[ tmp_77, tmp_65, tmp_53, tmp_70 ] , axis=[ tmp_70 ]), name='tensor_65_77_53_75_81' )
  tensor_65_22_23_77_83_81 = tvm.te.compute( (dim_65, dim_22, dim_23, dim_77, dim_83, dim_81), lambda tmp_65, tmp_22, tmp_23, tmp_77, tmp_83, tmp_81: tvm.te.sum( tensor_65_77_53_75_81[ tmp_65, tmp_77, tmp_53, tmp_75, tmp_81 ] * tensor_22_23_83_53_75[ tmp_22, tmp_23, tmp_83, tmp_53, tmp_75 ] , axis=[ tmp_75, tmp_53 ]), name='tensor_65_22_23_77_83_81' )
  tensor_9_41_65_86_81 = tvm.te.compute( (dim_9, dim_41, dim_65, dim_86, dim_81), lambda tmp_9, tmp_41, tmp_65, tmp_86, tmp_81: tvm.te.sum( tensor_65_22_23_77_83_81[ tmp_65, tmp_22, tmp_23, tmp_77, tmp_83, tmp_81 ] * tensor_9_41_22_23_86_77_83[ tmp_9, tmp_41, tmp_22, tmp_23, tmp_86, tmp_77, tmp_83 ] , axis=[ tmp_83, tmp_23, tmp_22, tmp_77 ]), name='tensor_9_41_65_86_81' )
  tensor_9_41_63_81 = tvm.te.compute( (dim_9, dim_41, dim_63, dim_81), lambda tmp_9, tmp_41, tmp_63, tmp_81: tvm.te.sum( tensor_9_41_65_86_81[ tmp_9, tmp_41, tmp_65, tmp_86, tmp_81 ] * tensor_86_65_63[ tmp_86, tmp_65, tmp_63 ] , axis=[ tmp_86, tmp_65 ]), name='tensor_9_41_63_81' )
  tensor_11_24_63_81_25 = tvm.te.compute( (dim_11, dim_24, dim_63, dim_81, dim_25), lambda tmp_11, tmp_24, tmp_63, tmp_81, tmp_25: tvm.te.sum( tensor_81_87_24_25[ tmp_81, tmp_87, tmp_24, tmp_25 ] * tensor_63_87_11[ tmp_63, tmp_87, tmp_11 ] , axis=[ tmp_87 ]), name='tensor_11_24_63_81_25' )
  tensor_11_24_9_41_25 = tvm.te.compute( (dim_11, dim_24, dim_9, dim_41, dim_25), lambda tmp_11, tmp_24, tmp_9, tmp_41, tmp_25: tvm.te.sum( tensor_11_24_63_81_25[ tmp_11, tmp_24, tmp_63, tmp_81, tmp_25 ] * tensor_9_41_63_81[ tmp_9, tmp_41, tmp_63, tmp_81 ] , axis=[ tmp_81, tmp_63 ]), name='tensor_11_24_9_41_25' )
  tensor_11_10_41_25 = tvm.te.compute( (dim_11, dim_10, dim_41, dim_25), lambda tmp_11, tmp_10, tmp_41, tmp_25: tvm.te.sum( tensor_11_24_9_41_25[ tmp_11, tmp_24, tmp_9, tmp_41, tmp_25 ] * tensor_9_10_24_25[ tmp_9, tmp_10, tmp_24, tmp_25 ] , axis=[ tmp_9, tmp_24 ]), name='tensor_11_10_41_25' )
  tensor_11_25 = tvm.te.compute( (dim_11, dim_25), lambda tmp_11, tmp_25: tvm.te.sum( tensor_11_10_41_25[ tmp_11, tmp_10, tmp_41, tmp_25 ] * tensor_41_10_11[ tmp_41, tmp_10, tmp_11 ] , axis=[ tmp_10, tmp_41 ]), name='tensor_11_25' )

  return [ tensor_47_8_9, tensor_7_8_22_23, tensor_39_40_46_47, tensor_46_6_7, tensor_27_30_31, tensor_26_27, tensor_37_38_44_45, tensor_33_38_39, tensor_35_36_42_43, tensor_43_0_1, tensor_42_50_51, tensor_51_0_14_15, tensor_34_48_49, tensor_49_50_12_13, tensor_55_48_68, tensor_58_55_67, tensor_62_69_73, tensor_73_56_14_15, tensor_72_56_74, tensor_67_57_62_72, tensor_74_84_16_17, tensor_1_2_16_17, tensor_71_60_18_19, tensor_3_4_18_19, tensor_80_79, tensor_80_58_64, tensor_82_76_52_54, tensor_59_76_77, tensor_66_83_20_21, tensor_5_6_20_21, tensor_61_75_22_23, tensor_53_83_61, tensor_70_75_81, tensor_77_65_53_70, tensor_81_87_24_25, tensor_63_87_11, tensor_41_10_11, tensor_9_10_24_25, tensor_86_65_63, tensor_31_40_41, tensor_26_28_29, tensor_29_30_32_33, tensor_32_36_37, tensor_44_2_3, tensor_28_34_35, tensor_68_69_12_13, tensor_52_84_71, tensor_45_4_5, tensor_79_78_86, tensor_64_78_85_59, tensor_85_57_82, tensor_54_60_66, tensor_11_25 ]

if __name__=="__main__":
  args = tvm_helper.parse_args()

  target = tvm.target.Target( tvm_helper.cpu_to_llvm( args.cpu ) )
  hardware_params = tvm.auto_scheduler.HardwareParams( target = target )
  dtype = args.dtype
  num_measure_trials = args.num_measure_trials
  timeout = args.timeout
  log_file = args.log_file

  einsum_str = "vIJ,HIWX,nouv,uGH,bef,ab,lmst,hmn,jkqr,rAB,qyz,zAOP,iwx,xyMN,ăwĐ,Ćăď,Ċđĕ,ĕĄOP,ĔĄĖ,ďąĊĔ,ĖĠQR,BCQR,ēĈST,DEST,Ĝě,ĜĆČ,ĞĘĀĂ,ćĘę,ĎğUV,FGUV,ĉėWX,āğĉ,Ēėĝ,ęčāĒ,ĝģYZ,ċģL,pKL,JKYZ,Ģčċ,fop,acd,degh,gkl,sCD,cij,ĐđMN,ĀĠē,tEF,ěĚĢ,ČĚġć,ġąĞ,ĂĈĎ->LZ"
  func = einsum_tree
  sizes = (16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 7, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16)

  tvm_helper.run_all( einsum_str,
                      func,
                      sizes,
                      dtype,
                      hardware_params,
                      target,
                      num_measure_trials,
                      timeout,
                      log_file )
