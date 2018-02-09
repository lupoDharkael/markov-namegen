cmake_minimum_required(VERSION 2.6)
project(markov)

add_executable(markov main.cpp)

install(TARGETS markov RUNTIME DESTINATION bin)
