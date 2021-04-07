add_library(paper_value_functions OBJECT
  experience_replay.h
  experience_replay.cc
  net_architectures.h
  net_architectures.cc
  net_data.h
  net_data.cc
  net_dl_evaluator.cc
  net_dl_evaluator.h
  particle.h
  particle.cc
  hand_table.h
  hand_table.cc
  solve_queue.h
  solve_queue.cc
  sparse_trunk.cc
  sparse_trunk.h
  subgame.h
  subgame.cc
  metrics.h
  metrics.cc
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
add_paper_executable(hand_table_test hand_table_test.cc)
add_paper_executable(sparse_trunk_test sparse_trunk_test.cc)
