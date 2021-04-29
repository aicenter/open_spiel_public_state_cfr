add_library(paper_value_functions OBJECT
  experience_replay.cc
  experience_replay.h
  hand_table.cc
  hand_table.h
  metrics.cc
  metrics.h
  net_architectures.cc
  net_architectures.h
  net_data.cc
  net_data.h
  net_dl_evaluator.cc
  net_dl_evaluator.h
  particle.cc
  particle.h
  particle_regeneration.cc
  particle_regeneration.h
  reusable_structs.cc
  reusable_structs.h
  solve_queue.cc
  solve_queue.h
  subgame.cc
  subgame.h
  subgame_factory.cc
  subgame_factory.h
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
add_test(hand_table_test hand_table_test)
add_paper_executable(particle_regeneration_test particle_regeneration_test.cc)
add_test(particle_regeneration_test particle_regeneration_test)
add_paper_executable(subgame_test subgame_test.cc)
add_test(subgame_test subgame_test)