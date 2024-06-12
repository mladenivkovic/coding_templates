#include "ParticleIdentifier.h"

// TODO: temporary

#include "Assertions.h"

// #include <iterator>
// #include <limits>
// #include <ranges>



#if !defined(AssignmentChecks) and !defined(noAssignmentChecks) \
  and PeanoDebug > 0
#define AssignmentChecks
#endif

toolbox::particles::assignmentchecks::internal::ParticleIdentifier::
  ParticleIdentifier(
    const std::string&                           particleName__,
    const tarch::la::Vector<Dimensions, double>& particleX__,
    const int                                    particleID__,
    const double                                 positionTolerance__
  ):
  particleName(particleName__),
  particleX(particleX__),
  particleID(particleID__),
  positionTolerance(positionTolerance__) {}

double toolbox::particles::assignmentchecks::internal::ParticleIdentifier::Precision = 1e-2;

bool toolbox::particles::assignmentchecks::internal::ParticleIdentifier::
  numericalEquals(const ParticleIdentifier& rhs) const {

  return (particleName == rhs.particleName) and (particleID == rhs.particleID)
         and tarch::la::equals(particleX, rhs.particleX, positionTolerance * Precision);
}

bool toolbox::particles::assignmentchecks::internal::ParticleIdentifier::
  operator<(const ParticleIdentifier& rhs) const {
  if (*this == rhs) {
    return false;
  }
  if (particleName < rhs.particleName) {
    return true;
  }
  if (particleName > rhs.particleName) {
    return false;
  }
  if (particleID < rhs.particleID)
    return true;
  if (particleID > rhs.particleID)
    return false;
  for (int d = 0; d < Dimensions; d++) {
    if (particleX(d) < rhs.particleX(d)) {
      return true;
    }
    if (particleX(d) > rhs.particleX(d)) {
      return false;
    }
  }
  assertion3(false, toString(), rhs.toString(), "cannot happen");
  return false;
}


std::string toolbox::particles::assignmentchecks::internal::ParticleIdentifier::
  toString() const {
  return "(" + particleName + ",ID=" + std::to_string(particleID) + ","
         + ::toString(particleX) + ")";
}


bool operator==(
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs,
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs
) {
  return lhs.numericalEquals(rhs);
}


bool operator!=(
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& lhs,
  const toolbox::particles::assignmentchecks::internal::ParticleIdentifier& rhs
) {
  return not lhs.numericalEquals(rhs);
}
