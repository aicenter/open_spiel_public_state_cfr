add_library(paper_value_functions OBJECT
  generate_data.h
  generate_data.cc
)

add_executable(generate_data_test generate_data_test.cc
  $<TARGET_OBJECTS:paper_value_functions> $<TARGET_OBJECTS:open_spiel_ortools> ${OPEN_SPIEL_OBJECTS})
target_link_libraries(generate_data_test ${ORTOOLS_LIBS} ${TORCH_LIBRARIES})

add_executable(value_functions_train_test_loop train_test_loop.cc
  $<TARGET_OBJECTS:paper_value_functions> $<TARGET_OBJECTS:open_spiel_ortools> ${OPEN_SPIEL_OBJECTS})
target_link_libraries(value_functions_train_test_loop ${ORTOOLS_LIBS} ${TORCH_LIBRARIES})

add_executable(value_fn_convergence value_fn_convergence.cc
  $<TARGET_OBJECTS:open_spiel_ortools> ${OPEN_SPIEL_OBJECTS})
target_link_libraries(value_fn_convergence
  absl::flags_parse absl::flags
  ${ORTOOLS_LIBS})
