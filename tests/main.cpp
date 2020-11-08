/**
 * @file main.cpp
 * @author Conrad Sanderson
 *
 * ensmallen is free software; you may redistribute it and/or modify it under
 * the terms of the 3-clause BSD license.  You should have received a copy of
 * the 3-clause BSD license along with ensmallen.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <iostream>
#define COOT_DEFAULT_BACKEND CUDA_BACKEND
#define COOT_USE_U64S64
#define ENS_PRINT_INFO
#define ENS_PRINT_WARN
#include <ensmallen.hpp>

//#define CATCH_CONFIG_MAIN  // catch.hpp will define main()
#define CATCH_CONFIG_RUNNER  // we will define main()
#include "catch.hpp"

int main(int argc, char** argv)
{
  coot::get_rt().init(true);

  /**
   * Uncomment these three lines if you want to test with different random seeds
   * each run.  This is good for ensuring that a test's tolerance is sufficient
   * across many different runs.
   */
  //size_t seed = std::time(NULL);
  //srand((unsigned int) seed);
  //arma::arma_rng::set_seed(seed);

  std::cout << "ensmallen version: " << ens::version::as_string() << std::endl;

  std::cout << "armadillo version: " << arma::arma_version::as_string() << std::endl;

  return Catch::Session().run(argc, argv);
}
