cmake_minimum_required(VERSION 3.5)
project(abb_irb1600_support)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch meshes urdf
  DESTINATION share/${PROJECT_NAME})

ament_package()