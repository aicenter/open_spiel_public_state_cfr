add_library(paper_value_functions OBJECT
  experience_replay.h
  experience_replay.cc
  generate_data.h
  generate_data.cc
  net_architectures.h
  net_data.h
  net_data.cc
  net_dl_evaluator.cc
  net_dl_evaluator.h
  range_table.h
  range_table.cc
  torch_utils.cc
  torch_utils.h
  train_eval.h
  train_eval.cc
  trunk.cc
  trunk.h
  )

# Delete -Wall -Werror in
# - /home/michal/Code/open_spiel/open_spiel/libtorch/libtorch/share/cmake/Caffe2/Caffe2Targets.cmake torch_cpu
# if warnings happen.
macro(add_paper_executable NAME SOURCES)
  add_executable(${NAME} ${SOURCES} $<TARGET_OBJECTS:paper_value_functions>
                                    $<TARGET_OBJECTS:open_spiel_ortools>
                                    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(${NAME} ${ORTOOLS_LIBS} ${TORCH_LIBRARIES})
endmacro()

add_paper_executable(train_eval_loop train_eval_loop.cc)
add_paper_executable(value_functions_train_eval_test train_eval_test.cc)
add_paper_executable(range_table_test range_table_test.cc)

add_paper_executable(value_fn_convergence value_fn_convergence.cc)
target_link_libraries(value_fn_convergence
                      absl::flags_parse absl::flags ${ORTOOLS_LIBS})
