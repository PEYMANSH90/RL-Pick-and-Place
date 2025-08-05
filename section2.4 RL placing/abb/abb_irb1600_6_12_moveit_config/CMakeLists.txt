cmake_minimum_required(VERSION 3.5)
project(abb_irb1600_6_12_moveit_config)

find_package(ament_cmake REQUIRED)


install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

ament_package()