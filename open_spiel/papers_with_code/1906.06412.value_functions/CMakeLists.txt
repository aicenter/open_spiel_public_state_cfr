add_library(paper_value_functions OBJECT
  generate_data.h
  generate_data.cc
  net_architectures.h
  net_batch.h
  net_dl_evaluator.cc
  net_dl_evaluator.h
  torch_utils.cc
  torch_utils.h
  train_eval.h
  train_eval.cc
  trunk.cc
  trunk.h
  )

add_executable(train_eval_loop train_eval_loop.cc
  $<TARGET_OBJECTS:paper_value_functions> $<TARGET_OBJECTS:open_spiel_ortools> ${OPEN_SPIEL_OBJECTS})
target_link_libraries(train_eval_loop ${ORTOOLS_LIBS} ${TORCH_LIBRARIES})

add_executable(value_functions_train_eval_test train_eval_test.cc
  $<TARGET_OBJECTS:paper_value_functions> $<TARGET_OBJECTS:open_spiel_ortools> ${OPEN_SPIEL_OBJECTS})
target_link_libraries(value_functions_train_eval_test ${ORTOOLS_LIBS} ${TORCH_LIBRARIES})

add_executable(value_fn_convergence value_fn_convergence.cc
  $<TARGET_OBJECTS:open_spiel_ortools> ${OPEN_SPIEL_OBJECTS})
target_link_libraries(value_fn_convergence
  absl::flags_parse absl::flags
  ${ORTOOLS_LIBS})
