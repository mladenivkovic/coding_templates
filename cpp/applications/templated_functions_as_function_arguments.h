// using functions as arguments for other functions

#pragma once

// #include <functional> // needed for std::function
/* #include <iostream> */


#include "PeanoPart.h"






// template <typename ParticleContainer>

/**
 * Pass a function as an argument which requires one templated argument
 */
template <typename ParticleContainer>
void call_function_with_one_templated_parameter(
  const ParticleContainer&    localParticles,
  bool (*workOnParticle)(hydroPart &)
) {

  for (auto* particle: localParticles) {
    if (workOnParticle(*particle)) particle->sayHello();
  }
}


